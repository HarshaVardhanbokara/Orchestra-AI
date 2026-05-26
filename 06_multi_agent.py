import os
import asyncio
from dotenv import load_dotenv

# 1. Load the API keys from the .env file FIRST
load_dotenv()

from typing import TypedDict, Annotated, Sequence
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

# 2. Define the Shared Memory (The Whiteboard)
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str

# Initialize the Graph
workflow = StateGraph(AgentState)

# 3. Define the Team Members
members = ["web_researcher", "file_manager"]
options = ["FINISH"] + members

# 4. Force Structured Output (The Routing Logic)
class Route(BaseModel):
    next: Literal["web_researcher", "file_manager", "FINISH"] = Field(
        description="The next worker to act, or FINISH if the task is complete."
    )

# Initialize the Boss
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# 5. The Supervisor's System Prompt
system_prompt = (
    "You are a strict technical supervisor managing a conversation between these workers: {members}. "
    "Given the user request, determine which worker needs to act next. "
    "The 'web_researcher' can search the internet for up-to-date information. "
    "The 'file_manager' can read local computer files using the Model Context Protocol. "
    "When the task is fully answered, or if no workers are needed, respond with FINISH."
)

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
    
    # 1. Initialize the live web search tool
    search_tool = DuckDuckGoSearchRun()
    
    # 2. Create the specialized internet agent
    # We pass it the search tool in a list
    worker_agent = create_react_agent(llm, tools=[search_tool])
    
    print("[Web Researcher] 🔍 Executing internet search...")
    # 3. Hand the conversation history to the agent
    result = await worker_agent.ainvoke({"messages": state["messages"]})
    
    # 4. Extract the final answer and pass it back
    final_answer = result["messages"][-1]
    
    return {"messages": [final_answer]}

async def file_manager_node(state: AgentState):
    print("\n[File Manager] 📂 Booting up MCP Server connection...")
    
    # Dynamically get the project directory so MCP knows where to look
    target_directory = os.path.abspath(os.getcwd())
    
    # Configure the npx Filesystem server
    server_config = {
        "npx": {
            "transport": "stdio",  # <--- ADD THIS LINE
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", target_directory],
        }
    }
    
    # --- THE FIX IS HERE ---
    # We no longer use 'async with'. We just initialize the client directly.
    client = MultiServerMCPClient(server_config)
    
    # Grab the file-reading tools dynamically
    mcp_tools = await client.get_tools()
    print(f"[File Manager] Successfully loaded {len(mcp_tools)} tools.")
    
    # Create a specialized worker agent on the fly
    worker_agent = create_react_agent(llm, tools=mcp_tools)
    
    print("[File Manager] 🔍 Executing task...")
    # Hand the entire conversation history to the worker so it knows what the user asked
    result = await worker_agent.ainvoke({"messages": state["messages"]})
    
    # Extract the worker's final answer and pass it back to the graph's shared whiteboard
    final_answer = result["messages"][-1]
    
    return {"messages": [final_answer]}

# 8. Build the Graph (Wiring the team together)
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("web_researcher", web_researcher_node)
workflow.add_node("file_manager", file_manager_node)

# The conversation ALWAYS starts with the boss
workflow.set_entry_point("supervisor")

# The boss decides who goes next based on the structured output
workflow.add_conditional_edges(
    "supervisor",
    lambda state: state["next"], 
    {
        "web_researcher": "web_researcher",
        "file_manager": "file_manager",
        "FINISH": END
    }
)

# After a worker finishes, the graph ends
workflow.add_edge("web_researcher", END)
workflow.add_edge("file_manager", END)

# Compile the machine!
graph = workflow.compile()
print("Graph Compiled Successfully!")

# 9. Test the Asynchronous Routing System
async def main():
    print("\n--- TESTING ASYNC MULTI-AGENT ROUTING ---")
    
    test_prompt = "What is the current price of Bitcoin today, and what is the latest news about it?"
    print(f"User: {test_prompt}")
    
    test_message = HumanMessage(content=test_prompt)
    
    # Execute the graph
    result = await graph.ainvoke({"messages": [test_message]})
    
    # Print the final output from the worker
    print("\n[Final Output]:")
    print(result["messages"][-1].content)
    
    print("\n[System] Graph execution finished cleanly.")

if __name__ == "__main__":
    asyncio.run(main())