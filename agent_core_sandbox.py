import os
import asyncio
import aiosqlite
from dotenv import load_dotenv

# 1. Load the API keys from the .env file FIRST
load_dotenv()

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from typing import TypedDict, Annotated, Sequence, List
import operator
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field
from typing import Literal
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_community.tools import DuckDuckGoSearchRun

# --- NEW IMPORTS FOR STRICT PYDANTIC TOOL ---
from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL
# --------------------------------------------

# 2. Define the Shared Memory (The Whiteboard)
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str

# Initialize the Graph
workflow = StateGraph(AgentState)

# 3. Define the Team Members
members = ["web_researcher", "file_manager","data_analyst"]
options = ["FINISH"] + members

# 4. Force Structured Output (The Routing Logic)
class Route(BaseModel):
    next: Literal["web_researcher", "file_manager", "data_analyst", "FINISH"] = Field(
        description="The next worker to act, or FINISH if the task is complete."
    )

# Initialize the Boss
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# 5. The Supervisor's System Prompt
system_prompt ="""You are an expert Autonomous Data Scientist and Financial Analyst.
        Your primary tool is 'python_executor'. 
    
        CRITICAL RULES FOR DATA ANALYSIS:
        1. If the user provides a CSV file path, you MUST use `pandas` to load it.
        2. Always print the results (e.g., `print(df['Revenue'].sum())`) so the terminal captures it.
        3. Provide your code as a list of strings (one string per line of code).
        4. ABSOLUTE RULE: You MUST use SINGLE QUOTES ('') for all strings inside your Python code. NEVER use double quotes (").
        5. ABSOLUTE RULE: DO NOT ATTEMPT TO PLOT OR VISUALIZE DATA (No matplotlib, no seaborn). You are running in a headless environment. You must summarize the data purely using text and math calculations.
        
        CRITICAL RULE FOR FINAL OUTPUT:
        After your tool successfully runs, write a text response summarizing the exact numbers.
        """

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="messages"),
    ("system", "Given the conversation above, who should act next? Select exactly one of: {options}")
]).partial(options=str(options), members=", ".join(members))

supervisor_chain = prompt | llm.with_structured_output(Route)

# 6. The Actual Supervisor Node Function
async def supervisor_node(state: AgentState):
    print("\n[Supervisor] Analyzing the request...")
    result = await supervisor_chain.ainvoke(state)
    print(f"[Supervisor] Routing task to -> {result.next.upper()}")
    return {"next": result.next}


# 7. Create the Worker Nodes
async def web_researcher_node(state: AgentState):
    print("\n[Web Researcher] 🌐 Booting up Web Search tools...")
    search_tool = DuckDuckGoSearchRun()
    worker_agent = create_react_agent(llm, tools=[search_tool])
    
    print("[Web Researcher] 🔍 Executing internet search...")
    result = await worker_agent.ainvoke({"messages": state["messages"]})
    final_answer = result["messages"][-1]
    return {"messages": [final_answer]}

async def file_manager_node(state: AgentState):
    print("\n[File Manager] 📂 Booting up MCP Server connection...")
    target_directory = os.path.abspath(os.getcwd())
    
    server_config = {
        "npx": {
            "transport": "stdio",
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", target_directory],
        }
    }
    
    client = MultiServerMCPClient(server_config)
    mcp_tools = await client.get_tools()
    print(f"[File Manager] Successfully loaded {len(mcp_tools)} tools.")
    
    worker_agent = create_react_agent(llm, tools=mcp_tools)
    
    print("[File Manager] 🔍 Executing task...")
    result = await worker_agent.ainvoke({"messages": state["messages"]})
    final_answer = result["messages"][-1]
    return {"messages": [final_answer]}

# --- THE FIX: STRICT PYDANTIC PYTHON TOOL ---
# --- THE BULLETPROOF PYDANTIC TOOL ---
repl = PythonREPL()

class PythonToolInput(BaseModel):
    # We force the AI to send a list of strings instead of one massive block
    code_lines: List[str] = Field(description="A list of python code lines to execute. MUST use SINGLE QUOTES ('') for strings inside the code. NEVER use double quotes.")

@tool("python_executor", args_schema=PythonToolInput)
def execute_python(code_lines: List[str]) -> str:
    """Executes python code in a REPL environment and returns the stdout."""
    try:
        # We manually join the lines back together into a valid Python script
        script = "\n".join(code_lines)
        
        # Print what the AI wrote to the terminal so you can monitor it!
        print(f"\n--- AI GENERATED SCRIPT ---\n{script}\n---------------------------\n")
        
        return repl.run(script)
    except Exception as e:
        return f"Code execution failed: {str(e)}"
# -------------------------------------------

async def data_analyst_node(state: AgentState):
    print("\n[Data Analyst] Processing Mathematical Query/ data request...")

    system_prompt ="""You are an expert Autonomous Data Scientist and Financial Analyst.
        Your primary tool is 'python_executor'. 
    
        CRITICAL RULES FOR DATA ANALYSIS:
        1. If the user provides a CSV file path, you MUST use `pandas` to load it.
        2. Always print the results (e.g., `print(df['Revenue'].sum())`) so the terminal captures it.
        3. Provide your code as a list of strings (one string per line of code).
        4. ABSOLUTE RULE: You MUST use SINGLE QUOTES ('') for all strings inside your Python code. NEVER use double quotes (").
        
        CRITICAL RULE FOR FINAL OUTPUT:
        After your tool successfully runs, write a text response summarizing the exact numbers.
        """

    worker_agent = create_react_agent(
        llm,
        tools = [execute_python], 
        prompt = system_prompt
    )

    print("[Data Analyst] 🔍 Executing task...")
    result = await worker_agent.ainvoke({"messages":state["messages"]})
    final_answer = result["messages"][-1]
    return {"messages": [final_answer]}

# 8. Build the Graph (Wiring the team together)
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("web_researcher", web_researcher_node)
workflow.add_node("file_manager", file_manager_node)
workflow.add_node("data_analyst", data_analyst_node)

workflow.set_entry_point("supervisor")

workflow.add_conditional_edges(
    "supervisor",
    lambda state: state["next"], 
    {
        "web_researcher": "web_researcher",
        "file_manager": "file_manager",
        "data_analyst": "data_analyst",
        "FINISH": END
    }
)

workflow.add_edge("web_researcher", END)
workflow.add_edge("file_manager", END)
workflow.add_edge("data_analyst", END)

# 9. Test the Asynchronous Routing System
async def main():
    print("\n--- TESTING ASYNC MULTI-AGENT ROUTING ---")
    
    test_prompt = input("Ask Multi-Agent System: ")
    print(f"User: {test_prompt}")
    
    test_message = HumanMessage(content=test_prompt)

    thread_config = {"configurable": {"thread_id": "audit_session_0"}}
    
    async with AsyncSqliteSaver.from_conn_string("cfo_memory.db") as memory:
        graph = workflow.compile(checkpointer=memory)
        # Execute the graph
        result = await graph.ainvoke({"messages": [test_message]},config=thread_config)
    
    print("\n[Final Output]:")
    print(result["messages"][-1].content)
    print("\n[System] Graph execution finished cleanly.")

if __name__ == "__main__":
    while True:
        asyncio.run(main())