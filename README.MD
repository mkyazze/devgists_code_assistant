## Code Assistant Server

A local FastAPI server that connects VS Code to local LLM models for code analysis and intelligent responses.

Read Tutorial at: https://substack.com/home/post/p-153166269
VS Code Extension: https://marketplace.visualstudio.com/items?itemName=DrMichaelKyazze.devgists-code-assistant

#### Requirements
```python
pip install llama-cpp-python fastapi "uvicorn[standard]"
```

#### Setup
- Step 1: Download a local LLM model (GGUF format)
- Step 2: Configure the model path

```python
from code_assistant import CodeAssistant

assistant = CodeAssistant(
    root_dir="./your_project_root", 
    model_path="/path/to/your/model.gguf"
)
```

#### Configuration
 - n_ctx: Context window size (default: 4096)
 - n_threads: CPU threads to use
 - Supported file extensions: `.py`, `.js`, `.ts`, `.java`, `.cpp`, `.h`, `.jsx`, `.tsx`

## API Endpoints

#### Analyze Code
- Endpoint: POST /analyze
- Description: Analyzes code and answers questions about specific code functionality.
- Example Request Body:
```json
{
     "file_path": "path/to/file",
     "question": "What does this code do?"
 }
```

#### Improve Code
- Endpoint: POST /improve
- Description: Provides suggestions for code improvements, best practices, and potential optimizations.
- Example Request Body:
```json
{
    "file_path": "path/to/file"
}
```

#### Search
- Endpoint: POST /search
- Description: Searches through the codebase for specific patterns or functionality.
- Example Request Body:
```json
{
    "query": "search term"
}
```

#### Modify
- Endpoint: POST /modify
- Description: Updates the content of a specified file with new code.
- Example Request Body:
```json
{
    "file_path": "path/to/file",
    "new_content": "updated code content"
}
```

#### Usage: Start the server
```python
python code_assistant.py
```

The server will run on http://localhost:8000
Use the DevGists VS Code extensions.
