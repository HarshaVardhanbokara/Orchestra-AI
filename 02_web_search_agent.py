import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain.agents import create_agent

# NEW: Import the pre-built web search tool
from langchain_community.tools import DuckDuckGoSearchRun

load_dotenv()

# 1. Your Custom Tool
@tool
def get_system_time() -> str:
    """Returns the current date and time of the system."""
    import datetime
    print("\n[SYSTEM LOG] >> Executing system time function...")
    return str(datetime.datetime.now())

def app():
    print("Initializing Web-Connected Agent...\n")
    
    # 2. The Model
    gemini_model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        temperature=0
    )
    
    # 3. Initialize the Web Search Tool
    search_tool = DuckDuckGoSearchRun()
    
    # 4. Create the Agent (Now with TWO tools)
    agent = create_agent(
        model=gemini_model,
        tools=[get_system_time, search_tool],
        system_prompt="You are a helpful and precise assistant. Use tools to find real-time data or search the web when necessary. Keep your final answer concise."
    )
    
    # 5. Run a prompt that REQUIRES the internet
    prompt = "Who won the most recent cricket T20 World Cup, and what is the exact date and time right now?"
    print(f"User: {prompt}")
    
    result = agent.invoke({
        "messages": [
            {"role": "user", "content": prompt}
        ]
    })
    
    print("\n--- FINAL OUTPUT ---")
    final_content = result["messages"][-1].content
    clean_text = final_content[0]["text"] if isinstance(final_content, list) else final_content
    print(clean_text)

if __name__ == "__main__":
    app()