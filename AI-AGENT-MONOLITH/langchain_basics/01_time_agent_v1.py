import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
# NEW: The modern v1.x import
from langchain.agents import create_agent

load_dotenv()

# 1. Your Tool
@tool
def get_system_time() -> str:
    """Returns the current date and time of the system."""
    import datetime
    print("\n[SYSTEM LOG] >> Python is executing the time function locally...")
    return str(datetime.datetime.now())

def app():
    print("Initializing Modern LangChain v1.x Agent...\n")
    
    # 2. The Model
    gemini_model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        temperature=0
    )
    
    # 3. Create the Agent (v1.x style)
    # We just pass the model, tools, and instructions directly.
    agent = create_agent(
        model=gemini_model,
        tools=[get_system_time],
        system_prompt="You are a precise data assistant. You must ALWAYS use tools to fetch real-time data. Return your final answer in a strict format: 'Date: [Value] | Time: [Value]'."
    )
    
    # 4. Run the system
    print("User: What is the exact date and time right now?")
    
    # NEW: In v1.x, we pass inputs as a list of messages
    result = agent.invoke({
        "messages": [
            {"role": "user", "content": "What is the exact date and time right now?"}
        ]
    })
    
    print("\n--- FINAL STRUCTURED OUTPUT ---")
    
    # Extract the raw content from the last message
    final_content = result["messages"][-1].content
    
    # If it's a list (multi-modal block), grab the text. If it's a string, just print it.
    clean_text = final_content[0]["text"] if isinstance(final_content, list) else final_content
    print(clean_text)

if __name__ == "__main__":
    app()