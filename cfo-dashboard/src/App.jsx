import { useState, useRef, useEffect } from 'react';
import './App.css'; // We will add a few lines of CSS next

function App() {
  const [messages, setMessages] = useState([]);
  const [inputText, setInputText] = useState('');
  const [selectedFile, setSelectedFile] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);

  const [sessionId] = useState(`session_${Date.now()}`);

  // Auto-scroll to the bottom when new messages arrive
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setSelectedFile(e.target.files[0]);
    }
  };

  const handleSendMessage = async (e) => {
    e.preventDefault();
    if (!inputText.trim() && !selectedFile) return;

    // 1. Add the user's message to the UI immediately
    const userMessage = {
      role: 'user',
      content: inputText,
      fileName: selectedFile ? selectedFile.name : null,
    };
    setMessages((prev) => [...prev, userMessage]);
    
    // 2. Prepare the payload
    const formData = new FormData();
    formData.append('message', inputText || "Please analyze the attached file.");
    formData.append('thread_id', sessionId); // Hardcoded for this showcase
    if (selectedFile) {
      formData.append('file', selectedFile);
    }

    // 3. Clear the inputs and show loading state
    setInputText('');
    setSelectedFile(null);
    setIsLoading(true);

    // 4. Fire the request to your FastAPI Backend
    try {
      const response = await fetch('http://localhost:8000/chat', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      // 5. Add the AI's response to the UI
      if (data.status === 'success') {
        setMessages((prev) => [
          ...prev,
          { role: 'ai', content: data.reply },
        ]);
      } else {
        setMessages((prev) => [
          ...prev,
          { role: 'error', content: `Error: ${data.reply}` },
        ]);
      }
    } catch (error) {
      console.error('API Error:', error);
      setMessages((prev) => [
        ...prev,
        { role: 'error', content: 'Failed to connect to the AI CFO Backend. Is the server running?' },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="chat-container">
      <header className="chat-header">
        <h1>AI CFO Dashboard</h1>
        <p>Powered by LangGraph & FastAPI</p>
      </header>

      <div className="chat-window">
        {messages.length === 0 ? (
          <div className="empty-state">
            <p>Upload a financial CSV and ask a question to begin.</p>
          </div>
        ) : (
          messages.map((msg, index) => (
            <div key={index} className={`message-wrapper ${msg.role}`}>
              <div className="message-bubble">
                {msg.fileName && (
                  <div className="attachment-badge">📎 {msg.fileName}</div>
                )}
                <span style={{ whiteSpace: 'pre-wrap' }}>{msg.content}</span>
              </div>
            </div>
          ))
        )}
        {isLoading && (
          <div className="message-wrapper ai">
            <div className="message-bubble loading">
              Thinking... (Executing AI Agents)
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      <form onSubmit={handleSendMessage} className="input-area">
        <div className="file-input-wrapper">
          <input
            type="file"
            id="file-upload"
            onChange={handleFileChange}
            onClick={(e) => { e.target.value = null; }} // <--- ADD THIS EXACT LINE
            accept=".csv,.txt,.json"
          />
          <label htmlFor="file-upload" className={selectedFile ? 'file-selected' : ''}>
            {selectedFile ? '📎 ' + selectedFile.name : '📎 Attach File'}
          </label>
        </div>
        
        <input
          type="text"
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          placeholder="Ask the Data Analyst to calculate revenue..."
          disabled={isLoading}
        />
        
        <button type="submit" disabled={isLoading || (!inputText.trim() && !selectedFile)}>
          Send
        </button>
      </form>
    </div>
  );
}

export default App;