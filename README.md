# AI-Agent-Monolith 🤖

An asynchronous, stateful multi-agent AI architecture built with LangGraph, LangChain, and Google Gemini. This system utilizes a centralized Supervisor agent to intelligently route natural language queries to specialized worker agents, enabling autonomous web research, local file system manipulation via the Model Context Protocol (MCP), and dynamic Python code execution.

## 🏗 Architecture Overview

The system operates on a state-machine architecture (`AgentState`) where all agents share a continuous memory stream. 

* **The Supervisor:** The central routing node. It analyzes the user's prompt and determines which worker agent is best suited for the task.
* **Web Researcher:** Equipped with DuckDuckGo search tools to autonomously query the live internet for up-to-date information, news, and real-time data.
* **File Manager:** Leverages a local Node.js MCP (Model Context Protocol) server to securely read, parse, and analyze local system files and directories.
* **Data Analyst:** An autonomous coding agent running inside a secure Python REPL sandbox. It can write, execute, and debug Python code (including `pandas` operations) to solve complex mathematical queries and analyze local datasets on the fly.

## 🛠 Tech Stack

* **Core Frameworks:** LangChain (v1.0+), LangGraph
* **LLM:** Google Gemini 2.5 Flash (`langchain-google-genai`)
* **Local Tooling:** Model Context Protocol (MCP) via `@modelcontextprotocol/server-filesystem`, Python REPL (`langchain-experimental`)
* **Search Integration:** DuckDuckGo Search (`langchain-community`)
* **Environment Management:** `python-dotenv`, `asyncio`

## 🚀 Getting Started

### Prerequisites
* Python 3.10+
* Node.js (Required for `npx` to boot the local MCP filesystem server)
* A valid Google Gemini API Key

### Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/AI-AGENT-MONOLITH.git](https://github.com/yourusername/AI-AGENT-MONOLITH.git)
   cd AI-AGENT-MONOLITH

   Create and activate a virtual environment:

Bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
Install the dependencies:
(Ensure your virtual environment is active before running this)

Bash
pip install -r requirements.txt
Set up Environment Variables:
Create a .env file in the root directory and add your API keys:

Code snippet
GEMINI_API_KEY=your_actual_api_key_here
💻 Usage
Make sure your virtual environment is active, then execute the desired script:

1. Run the Full Multi-Agent Orchestrator:
Boots up the Supervisor and dynamically routes tasks between the Web Researcher, File Manager, and Data Analyst.

Bash
python 06_multi_agent.py
2. Test the Standalone Data Analyst Sandbox:
A localized testing environment to verify the autonomous Python code execution and math reasoning engine.

Bash
python 07_data_analyst_sandbox.py
🔮 Future Roadmap
API Deployment: Wrap the LangGraph architecture in FastAPI to serve the agents as backend microservices for React frontend applications.

LLMOps Integration: Implement LangSmith for full telemetry tracking, latency monitoring, and token cost analysis.

RAG Capabilities: Integrate a vector database to allow the agents to perform Retrieval-Augmented Generation on large, custom document repositories.

📄 License
This project is licensed under the MIT License.