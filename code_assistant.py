import os
import glob
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException
import uvicorn
from pydantic import BaseModel
from llama_cpp import Llama
import logging
import multiprocessing

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CodeAssistant:
    def __init__(self, root_dir: str, model_path: str):
        """
        Initialize the code assistant with a root directory and Llama model.
        
        Args:
            root_dir: Root directory of the project
            model_path: Path to the Llama model file
        """
        self.root_dir = root_dir
        self.file_cache: Dict[str, str] = {}
        self.supported_extensions = {'.py', '.js', '.ts', '.java', '.cpp', '.h', '.jsx', '.tsx', '.sql'}
        
        # Initialize Llama
        try:
            self.llm = Llama(
                model_path=model_path,
                n_ctx=4096,  # Context window
                n_threads=max(4, multiprocessing.cpu_count() - 2),
                temperature=0.2  # Lower for faster, more focused responses
            )
            logger.info("Llama model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Llama model: {str(e)}")
            raise

    def scan_codebase(self) -> None:
        """Scan and cache all supported code files in the project."""
        for ext in self.supported_extensions:
            pattern = os.path.join(self.root_dir, f'**/*{ext}')
            files = glob.glob(pattern, recursive=True)
            
            for file_path in files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        self.file_cache[file_path] = f.read()
                    logger.info(f"Cached file: {file_path}")
                except Exception as e:
                    logger.error(f"Error reading {file_path}: {str(e)}")
                    
    def search_code(self, query: str) -> List[Dict[str, str]]:
        """
        Search through cached codebase for the query.
        
        Args:
            query: Search query string
            
        Returns:
            List of dictionaries containing file paths and relevant code snippets
        """
        results = []
        for file_path, content in self.file_cache.items():
            if query.lower() in content.lower():
                results.append({
                    'file': file_path,
                    'content': content
                })
        return results
    
    def modify_file(self, file_path: str, new_content: str) -> bool:
        """
        Modify a file with new content.
        
        Args:
            file_path: Path to the file to modify
            new_content: New content to write to the file
            
        Returns:
            Boolean indicating success
        """
        try:
            # Write new content to file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            # Update the cache
            self.file_cache[file_path] = new_content
            logger.info(f"Successfully modified file: {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error modifying {file_path}: {str(e)}")
            return False
    
    def get_llm_response(self, prompt: str, max_tokens: int = 500) -> str:
        """
        Get a response from the Llama model with system message.
        
        Args:
            prompt: Input prompt for the model
            max_tokens: Maximum tokens in response
            
        Returns:
            Model's response as a string
        """
        try:
            # Create a system message that instructs the model
            system_message = "You are a helpful coding assistant. Always provide direct answers without including the original prompt instructions in your response."
            
            # Combine system message with user prompt
            full_prompt = f"<system>{system_message}</system>\n\n<user>{prompt}</user>\n\n<assistant>"
            
            response = self.llm(
                full_prompt,
                max_tokens=max_tokens,
                temperature=0.7,
                stop=["</assistant>"]  # Stop generating at the end of the assistant's response
            )
            return response['choices'][0]['text'].strip()
        except Exception as e:
            logger.error(f"Error getting LLM response: {str(e)}")
            raise
    

    def analyze_code(self, prompt: str) -> str:
        """
        Analyze code using the LLM.
        
        Args:
            prompt: The complete prompt including code and question
            
        Returns:
            LLM's analysis
        """
        return self.get_llm_response(prompt, max_tokens=1000)
    

    def suggest_improvements(self, code: str) -> str:
        """
        Suggest improvements for the given code.
        
        Args:
            code: Code to improve
            
        Returns:
            Structured improvement suggestions
        """
        prompt = f"""Code to review:
        ```
        {code}
        ```
        
        Please analyze this code and provide specific, actionable improvements for:
        - Code quality and readability
        - Performance optimizations
        - Security considerations
        - Error handling
        - Testing suggestions
        """
        
        return self.get_llm_response(prompt, max_tokens=1500)

# FastAPI server setup
app = FastAPI()

# Pydantic models for request/response
class CodeAnalysisRequest(BaseModel):
    file_path: str
    question: str

class CodeModificationRequest(BaseModel):
    file_path: str
    new_content: str

class CodeSearchRequest(BaseModel):
    query: str

# Initialize CodeAssistant
assistant = CodeAssistant(
    root_dir="/Users/drkyazze/Documents/CODE_TO_WORK_WITH_XYZ", # Change / replace with root directory of the code you want to analyze
    model_path="/Users/drkyazze/.cache/lm-studio/models/TheBloke/CodeLlama-7B-Instruct-GGUF/codellama-7b-instruct.Q4_0.gguf")

@app.on_event("startup")
async def startup_event():
    """Scan codebase when server starts"""
    assistant.scan_codebase()

@app.post("/analyze")
async def analyze_code(request: CodeAnalysisRequest):
    """Endpoint to analyze code"""
    try:
        if request.file_path not in assistant.file_cache:
            logger.error(f"File not found in cache: {request.file_path}")
            raise HTTPException(status_code=404, detail="File not found")
        
        code = assistant.file_cache[request.file_path]
        logger.info(f"Analyzing file: {request.file_path}")
        logger.info(f"Code preview: {code[:100]}...")
        
        prompt = f"""Code to analyze:
        ```
        {code}
        ```
        
        Question: {request.question}"""

        analysis = assistant.analyze_code(prompt)
        return {"analysis": analysis}
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/improve")
async def improve_code(request: CodeAnalysisRequest):
    """Endpoint to get code improvements"""
    try:
        if request.file_path not in assistant.file_cache:
            raise HTTPException(status_code=404, detail="File not found")
        
        code = assistant.file_cache[request.file_path]
        improvements = assistant.suggest_improvements(code)
        return {"improvements": improvements}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/search")
async def search_code(request: CodeSearchRequest):
    """Endpoint to search codebase"""
    try:
        results = assistant.search_code(request.query)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/modify")
async def modify_file(request: CodeModificationRequest):
    """Endpoint to modify code files"""
    try:
        success = assistant.modify_file(request.file_path, request.new_content)
        return {"success": success}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def start_server():
    """Start the FastAPI server"""
    uvicorn.run(app, host="127.0.0.1", port=8000)

if __name__ == "__main__":
    start_server()
