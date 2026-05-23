import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_community.tools import DuckDuckGoSearchRun
# NEW: Import the PDF Loader
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()

@tool
def get_system_time() -> str:
    """Returns the current date and time of the system."""
    import datetime
    return str(datetime.datetime.now())

# NEW: The PDF Reader Tool
@tool
def read_local_pdf(file_path: str) -> str:
    """Reads the text from a local PDF file. Provide the exact file name like 'sample.pdf'."""
    print(f"\n[SYSTEM LOG] >> Opening and reading PDF: {file_path}...")
    try:
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        
        # Combine all the pages into one giant string of text
        full_text = "\n".join([page.page_content for page in pages])
        
        # We limit to first 10,000 characters just to ensure we don't overload the context window for this test
        return full_text[:10000] 
    except Exception as e:
        return f"Error reading PDF: {str(e)}"

def app():
    print("Initializing Multi-Modal Agent (Web + PDF + Time)...\n")
    
    gemini_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    search_tool = DuckDuckGoSearchRun()
    
    # We now pass THREE tools to the agent
    agent = create_agent(
        model=gemini_model,
        tools=[get_system_time, search_tool, read_local_pdf],
        system_prompt="You are a highly capable data assistant. You can browse the web, check the time, and read local PDF documents. Always verify the file name before reading a document."
    )
    
    chat_history = []
    print("Agent is online! Try asking it to summarize 'sample.pdf'. (Type 'quit' to stop)\n")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            break
            
        chat_history.append({"role": "user", "content": user_input})
        
        result = agent.invoke({"messages": chat_history})
        
        final_content = result["messages"][-1].content
        clean_text = final_content[0]["text"] if isinstance(final_content, list) else final_content
        
        print(f"\nAI: {clean_text}\n")
        chat_history.append({"role": "assistant", "content": clean_text})

if __name__ == "__main__":
    app()