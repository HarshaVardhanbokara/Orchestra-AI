import os
import asyncio
from dotenv import load_dotenv

# Load API Keys from your .env file
load_dotenv()

from typing import TypedDict, Annotated, Sequence
import operator
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langchain_experimental.tools import PythonREPLTool

# Initialize the Google Gemini LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# 1. Define the shared graph state memory structure
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]

# 2. Define the execution node for the Data Analyst
async def analyst_node(state: AgentState):
    print("\n[Data Analyst] 📊 Analyzing the request...")
    
    # Initialize the REPL tool securely inside the local node execution loop
    repl_tool = PythonREPLTool()
    
    # Create the internal ReAct engine using the prompt parameter
    worker_agent = create_react_agent(
        llm, 
        tools=[repl_tool],
        prompt="You are an expert data analyst. Use Python to solve the user's request. Always output your code or its prints so the user can verify it."
    )
    
    # Process the message trail
    result = await worker_agent.ainvoke({"messages": state["messages"]})
    return {"messages": result["messages"]}

# 3. Construct and compile the state graph
workflow = StateGraph(AgentState)
workflow.add_node("data_analyst", analyst_node)
workflow.set_entry_point("data_analyst")
workflow.add_edge("data_analyst", END)

graph = workflow.compile()
print("Graph Compiled Successfully!")

# 4. Asynchronous entry point for execution testing
async def main():
    print("\n--- TESTING STANDALONE DATA ANALYST ---")
    test_prompt = "Calculate the square root of 84,592 and multiply the result by 14.5. Show your code."
    
    test_message = HumanMessage(content=test_prompt)
    result = await graph.ainvoke({"messages": [test_message]})
    
    print("\n[Final Output]:")
    print(result["messages"][-1].content)
    print("\n[System] Execution finished cleanly.")

if __name__ == "__main__":
    asyncio.run(main())