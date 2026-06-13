from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
import uvicorn
import os
import shutil

# 1. Import 'workflow' from your newly renamed file
from agent_core import workflow 

app = FastAPI(title="AI CFO Backend Production API")

# Enable CORS so your React frontend can talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/chat")
async def chat_endpoint(
    message: str = Form(...),
    thread_id: str = Form(...),
    file: UploadFile = File(None)
):
    # Ensure the uploads directory exists
    os.makedirs("uploads", exist_ok=True)
    
    # FIX: Initialize file_path so the cleanup block doesn't crash if there's no file!
    file_path = None 
        
    if file:
        # Use an absolute path so the Python REPL can always find it
        abs_dir = os.path.abspath("uploads")
        file_path = os.path.join(abs_dir, file.filename)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
                
        safe_path = file_path.replace("\\", "/")
        message = f"{message}\n\n[System Note: The user uploaded a file located at {safe_path}. Use this absolute path for your analysis.]"

    thread_config = {"configurable": {"thread_id": thread_id}}
    response_data = None
    
    try:
        # 2. Open the memory connection and compile the workflow dynamically per request!
        async with AsyncSqliteSaver.from_conn_string("cfo_memory.db") as memory:
            graph = workflow.compile(checkpointer=memory)
            
            # Invoke the graph
            result = await graph.ainvoke({"messages": [{"role": "user", "content": message}]}, config=thread_config)
            
        # --- NEW SMARTER MESSAGE EXTRACTION ---
        final_text = "Error: AI did not return a valid text response. Check the terminal."
        
        # 1. Grab the ABSOLUTE last message in the entire graph state
        last_message = result["messages"][-1]
        msg_type = last_message.get("type", "") if isinstance(last_message, dict) else getattr(last_message, "type", "")
        
        # 2. Check if it's an AI message. If it's a User message, the Supervisor routed to FINISH without answering!
        if msg_type in ["ai", "assistant", "AIMessageChunk"]:
            raw_content = last_message.get("content", "") if isinstance(last_message, dict) else getattr(last_message, "content", "")
            
            # Print the ENTIRE message object to the terminal to reveal hidden tool calls/errors
            print(f"\n--- FULL AI MESSAGE DEBUG ---\n{last_message}\n----------------------------\n")
            
            # Extract the text safely
            if isinstance(raw_content, str) and raw_content.strip():
                final_text = raw_content.strip()
            elif isinstance(raw_content, list):
                text_blocks = [b["text"] for b in raw_content if isinstance(b, dict) and "text" in b]
                if text_blocks:
                    final_text = "\n".join(text_blocks)
                else:
                    final_text = str(raw_content) # Absolute fallback
        else:
            # The AI stayed silent. We gracefully inform the user.
            final_text = "[System] The Supervisor analyzed the request and ended the task without needing to generate a new answer."

        response_data = {"status": "success", "reply": final_text}
        # --------------------------------------
        
    except Exception as e:
        response_data = {"status": "error", "reply": str(e)}

    # Clean up temp files is DISABLED so the AI can remember them for follow-up questions!
    # if file_path and os.path.exists(file_path):
    #     os.remove(file_path)

    print(f"\n--- DEBUG: SENDING TO REACT ---\n{response_data}\n-------------------\n")
    return response_data

if __name__ == "__main__":
    print("Starting AI CFO Production API...")
    uvicorn.run(app, host="0.0.0.0", port=8000)