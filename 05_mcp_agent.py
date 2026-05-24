import asyncio
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent

# NEW: The correct, updated class name!
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()

async def app():
    print("Initializing Model Context Protocol (MCP) Server...\n")
    
    gemini_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    
    # Make sure this path points to your actual folder
    # This dynamically grabs the absolute path of the folder you are running the script in
    target_directory = os.path.abspath(os.path.join(os.getcwd(), ".."))
    
    print(f"[SYSTEM] Booting up npx Filesystem Server for {target_directory}...")
    
    # We initialize the client with a configuration dictionary
    client = MultiServerMCPClient({
        "my_local_desktop": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", target_directory],
            "transport": "stdio"
        }
    })
    
    # The agent dynamically asks the npx server for its tools
    mcp_tools = await client.get_tools()
    print(f"\n[SYSTEM] Successfully loaded {len(mcp_tools)} tools from the MCP Server!")
    
    # We build the LangGraph agent
    agent = create_react_agent(gemini_model, tools=mcp_tools)
    
    print("\nMCP Agent is online! Try asking: 'What files are in this folder?' (Type 'quit' to stop)\n")
    
    chat_history = []
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            print("Shutting down...")
            break
            
        chat_history.append(("human", user_input))
        
        result = await agent.ainvoke({"messages": chat_history})
        
        clean_text = result["messages"][-1].content
        print(f"\nAI: {clean_text}\n")
        
        chat_history.append(("assistant", clean_text))

if __name__ == "__main__":
    asyncio.run(app())