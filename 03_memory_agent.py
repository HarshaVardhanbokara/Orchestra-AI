import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_community.tools import DuckDuckGoSearchRun

load_dotenv()

# 1. Your Custom Tool
@tool
def get_system_time() -> str:
    """Returns the current date and time of the system."""
    import datetime
    return str(datetime.datetime.now())

def app():
    print("Initializing Stateful Memory Agent...\n")
    
    gemini_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    search_tool = DuckDuckGoSearchRun()
    
    agent = create_agent(
        model=gemini_model,
        tools=[get_system_time, search_tool],
        system_prompt="You are a helpful assistant. Use tools if needed. Keep answers concise."
    )
    
    # --- THE MEMORY UPGRADE ---
    # We create a blank list to store the conversation history
    chat_history = []
    
    print("Chatbot is live! (Type 'exit' or 'quit' to stop)\n")
    
    # The Infinite Conversation Loop
    while True:
        user_input = input("You: ")
        
        if user_input.lower() in ['exit', 'quit']:
            print("Shutting down...")
            break
            
        # 1. Add the new user question to the memory list
        chat_history.append({"role": "user", "content": user_input})
        
        # 2. Pass the ENTIRE memory list to the agent, not just the latest question
        result = agent.invoke({
            "messages": chat_history
        })
        
        # 3. Extract the final answer
        final_content = result["messages"][-1].content
        clean_text = final_content[0]["text"] if isinstance(final_content, list) else final_content
        
        print(f"\nAI: {clean_text}\n")
        
        # 4. Save the AI's answer back into the memory list so it remembers it for next time
        chat_history.append({"role": "assistant", "content": clean_text})

if __name__ == "__main__":
    app()