# ml_copilot_agent/api_server.py

import os
import logging
from fastapi import FastAPI, HTTPException, Body, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field # Ensure pydantic is installed (comes with fastapi)
from typing import Optional, List, Dict, Any
import requests # For Ollama check

# Import logic from workflow_logic
from . import workflow_logic

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- FastAPI App ---
app = FastAPI(title="ML Copilot Agent API")

# --- CORS Middleware ---
# Allow requests from your React frontend development server and potentially production URL
origins = [
    "http://localhost:3000", # Default React dev server
    "http://localhost:5173", # Default Vite dev server
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
    # Add the origin where your packaged app will be served if different,
    # though often file:// origins or localhost are used.
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods (GET, POST, etc.)
    allow_headers=["*"], # Allows all headers
)

# --- Pydantic Models for Request/Response Body ---

class PathRequest(BaseModel):
    path: str

class FileListResponse(BaseModel):
    files: List[str]
    current_directory: str

class ColumnsResponse(BaseModel):
    columns: List[str]

class SetDirectoryResponse(BaseModel):
    success: bool
    message: str
    path: str

class LLMConfigRequest(BaseModel):
    provider: str = Field(..., description="LLM provider ('openai' or 'ollama')")
    api_key: Optional[str] = Field(None, description="API Key (only for 'openai')")
    api_model: Optional[str] = Field("gpt-4o", description="Model name for OpenAI")
    ollama_model: Optional[str] = Field("llama3", description="Model name for Ollama")
    ollama_endpoint: Optional[str] = Field("http://localhost:11434", description="Ollama server endpoint")

class ConfigResponse(BaseModel):
    success: bool
    message: str

class OllamaModel(BaseModel):
    name: str
    # Add other fields from Ollama API if needed (modified_at, size, etc.)

class OllamaModelsResponse(BaseModel):
    models: List[OllamaModel]

class PreprocessRequest(BaseModel):
    dataset_path: str
    target_column: str
    save_path: Optional[str] = 'data/preprocessed_data.csv'
    columns_to_drop: Optional[List[str]] = None
    additional_instructions: Optional[str] = None

class TrainRequest(BaseModel):
    data_path: str
    target_column: str
    model_save_path: Optional[str] = 'models/model.pkl'
    task_type: str = "classification" # Add task_type
    additional_instructions: Optional[str] = None

class EvaluateRequest(BaseModel):
    data_path: str
    target_column: str
    model_path: str
    evaluation_save_path: Optional[str] = 'results/evaluation.txt'
    task_type: str = "classification" # Add task_type
    additional_instructions: Optional[str] = None # Added

class PlotRequest(BaseModel):
    plot_type: str # 'results' or 'data'
    data_file_path: str
    target_column: Optional[str] = None # Added
    plot_save_dir: Optional[str] = 'plots'
    additional_instructions: Optional[str] = None

class CustomInstructionRequest(BaseModel):
    instruction: str

class AutoPilotRequest(BaseModel):
    data_path: str
    target_column: str
    task_type: str
    iterations: int = Field(1, ge=1) # Ensure at least 1 iteration
    plot_save_dir: Optional[str] = 'plots'

class AgentResponse(BaseModel):
    message: str # Raw response from agent or error message

class EvaluateResponse(AgentResponse):
     results: Optional[Dict[str, Any]] = None # Parsed metrics

class PlotResponse(AgentResponse):
     plot_path: Optional[str] = None # Absolute path to the generated plot

class AutoPilotResponse(AgentResponse):
     plot_path: Optional[str] = None # Absolute path to the final box plot

class StatusResponse(BaseModel):
    status: str
    message: str
    working_directory: str
    config_status: Dict[str, str] # llm: 'ok'|'error'|'pending', cwd: 'ok'|'error'|'pending'

class ProjectListResponse(BaseModel):
    projects: List[str]

# --- API Endpoints ---

@app.get("/api/status", response_model=StatusResponse)
async def get_status():
    """Returns the current status and configuration of the backend."""
    # Determine config status more accurately if possible
    llm_status = 'pending'
    if workflow_logic.agent and workflow_logic.Settings.llm:
        llm_status = 'ok'
    # Add more checks, e.g., try a simple LLM call if needed to confirm 'ok'

    cwd_status = 'ok' # Assume ok if server is running, refine if needed

    return StatusResponse(
        status="idle", # Or 'busy' if processing a long task
        message="ML Copilot Agent Backend is running.",
        working_directory=workflow_logic.current_working_directory,
        config_status={"llm": llm_status, "cwd": cwd_status}
    )

# --- Configuration Endpoints ---

@app.post("/api/set_working_directory", response_model=SetDirectoryResponse)
async def set_cwd(request: PathRequest):
    """Sets the agent's working directory."""
    success = workflow_logic.set_working_directory(request.path)
    if success:
        return SetDirectoryResponse(
            success=True,
            message=f"Working directory set to {workflow_logic.current_working_directory}",
            path=workflow_logic.current_working_directory
        )
    else:
        # Log the error via the logic function
        raise HTTPException(status_code=400, detail=f"Failed to set working directory to {request.path}. Check logs.")

@app.get("/api/ollama_models", response_model=OllamaModelsResponse)
async def get_ollama_models(ollama_endpoint: str = Query("http://localhost:11434", description="Ollama server endpoint URL")):
    """Gets the list of models available from the local Ollama server."""
    logger.info(f"Querying Ollama models at: {ollama_endpoint}")
    try:
        # Use requests library to query Ollama API
        response = requests.get(f"{ollama_endpoint}/api/tags", timeout=10) # Add timeout
        response.raise_for_status() # Raise exception for bad status codes
        data = response.json()
        # Ensure the response structure is as expected
        if "models" not in data or not isinstance(data["models"], list):
             logger.error(f"Unexpected response format from Ollama: {data}")
             raise HTTPException(status_code=500, detail="Unexpected response format from Ollama.")

        # Pydantic will validate the structure based on OllamaModel
        return OllamaModelsResponse(models=[OllamaModel(**m) for m in data["models"]])
    except requests.exceptions.ConnectionError:
        logger.error(f"Connection error querying Ollama at {ollama_endpoint}.")
        raise HTTPException(status_code=503, detail=f"Could not connect to Ollama server at {ollama_endpoint}. Is it running?")
    except requests.exceptions.Timeout:
        logger.error(f"Timeout querying Ollama at {ollama_endpoint}.")
        raise HTTPException(status_code=504, detail=f"Timeout connecting to Ollama server at {ollama_endpoint}.")
    except requests.exceptions.RequestException as e:
        logger.error(f"Error querying Ollama models: {e}")
        raise HTTPException(status_code=500, detail=f"Error querying Ollama models: {e}")
    except Exception as e:
        logger.error(f"Unexpected error fetching Ollama models: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")


@app.post("/api/set_llm_config", response_model=ConfigResponse)
async def set_llm_config(request: LLMConfigRequest):
    """Configures the LLM provider and model for the agent."""
    try:
        # Convert pydantic model to dict for the logic function
        config_dict = request.model_dump()

        # *** SECURITY NOTE ***:
        # In a real production app, the API key should NOT be passed directly like this.
        # It should be read from a secure source (env variables, secrets manager)
        # on the backend based on user identity or configuration.
        # This implementation assumes the key passed is handled securely by configure_agent.
        if config_dict.get("provider") == "openai" and not config_dict.get("api_key"):
             # If key isn't passed, try reading from environment as fallback
             env_key = os.getenv("OPENAI_API_KEY")
             if env_key:
                 logger.info("Using OpenAI API key from environment variable.")
                 config_dict["api_key"] = env_key
             else:
                 logger.warning("OpenAI provider selected but no API key provided in request or environment.")
                 # Allow configuration without key, but agent might fail later
                 # raise HTTPException(status_code=400, detail="OpenAI API Key is required but not provided.")

        workflow_logic.configure_agent(config_dict)
        return ConfigResponse(success=True, message=f"LLM configuration updated to use {request.provider}")
    except ValueError as e: # Catch specific errors from configure_agent
         logger.error(f"LLM Configuration Error: {e}")
         raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to set LLM config: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error setting LLM config: {e}")


# --- File/Data Endpoints ---

@app.get("/api/files", response_model=FileListResponse)
async def list_files(path: str = Query(".", description="Directory path relative to working directory")):
    """Lists files and directories in the specified path."""
    try:
        files = await workflow_logic.list_files_in_path(path)
        # Return the path requested, which might differ from CWD if browsing subdirs
        return FileListResponse(files=files, current_directory=path)
    except Exception as e:
        logger.error(f"Error listing files for path '{path}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error listing files: {e}")

@app.get("/api/get_columns", response_model=ColumnsResponse)
async def get_columns(file_path: str = Query(..., description="Path to the CSV file relative to working directory")):
    """Gets column names from a CSV file."""
    try:
        columns = await workflow_logic.get_csv_columns(file_path)
        return ColumnsResponse(columns=columns)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e: # Catch errors from reading columns
         raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting columns for '{file_path}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error getting columns: {e}")


# --- ML Task Endpoints ---

@app.post("/api/preprocess", response_model=AgentResponse)
async def run_preprocess(request: PreprocessRequest):
    """Triggers the data preprocessing step."""
    try:
        response_msg = await workflow_logic.run_preprocessing_step(
            dataset_path=request.dataset_path,
            target_column=request.target_column,
            save_path=request.save_path or 'data/preprocessed_data.csv',
            columns_to_drop=request.columns_to_drop,
            additional_instructions=request.additional_instructions
        )
        return AgentResponse(message=response_msg)
    except Exception as e:
        logger.error(f"Error during /preprocess: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error during preprocessing: {e}")

@app.post("/api/train", response_model=AgentResponse)
async def run_train(request: TrainRequest):
    """Triggers the model training step."""
    try:
        response_msg = await workflow_logic.run_training_step(
            data_path=request.data_path,
            target_column=request.target_column, # Pass target column
            model_save_path=request.model_save_path or 'models/model.pkl',
            task_type=request.task_type, # Pass task type
            additional_instructions=request.additional_instructions
        )
        return AgentResponse(message=response_msg)
    except Exception as e:
        logger.error(f"Error during /train: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error during training: {e}")

@app.post("/api/evaluate", response_model=EvaluateResponse)
async def run_evaluate(request: EvaluateRequest):
    """Triggers the model evaluation step."""
    try:
        # The logic function now returns a dictionary
        result_dict = await workflow_logic.run_evaluation_step(
            data_path=request.data_path,
            target_column=request.target_column, # Pass target column
            model_path=request.model_path,
            evaluation_save_path=request.evaluation_save_path or 'results/evaluation.txt',
            task_type=request.task_type, # Pass task type
            additional_instructions=request.additional_instructions # Pass instructions
        )
        return EvaluateResponse(message=result_dict["message"], results=result_dict["results"])
    except Exception as e:
        logger.error(f"Error during /evaluate: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error during evaluation: {e}")

@app.post("/api/plot", response_model=PlotResponse)
async def run_plot(request: PlotRequest):
    """Triggers the plotting step."""
    try:
        # The logic function now returns a dictionary
        result_dict = await workflow_logic.run_plotting_step(
            plot_type=request.plot_type,
            data_file_path=request.data_file_path,
            target_column=request.target_column, # Pass target column
            plot_save_dir=request.plot_save_dir or 'plots',
            additional_instructions=request.additional_instructions
        )
        # Check if plot_path exists and is valid before returning
        # --- Start of potentially problematic indentation ---
        plot_path = result_dict.get("plot_path") # <-- Ensure this line has correct indentation relative to the 'try' block
        if plot_path and not os.path.exists(plot_path):
            logger.warning(f"Plot path reported but not found: {plot_path}")
            # Decide how to handle: return None, raise error, or return path anyway?
            # Returning None for now if file doesn't exist.
            plot_path = None
        # --- End of potentially problematic indentation ---

        # TODO: Need a way to serve the plot file.
        # Option 1: Return the absolute path and have Electron/Tauri load it via file://
        # Option 2: Add a static file serving endpoint in FastAPI and return a URL /plots/filename.png
        # Option 3: Read the image file and return base64 data (less ideal for large images)
        # Sticking with Option 1 (absolute path) for now, assuming Electron/Tauri context.
        # If running purely as a web app, Option 2 would be needed.

        return PlotResponse(message=result_dict["message"], plot_path=plot_path) # <-- Ensure this line has correct indentation

    except Exception as e:
        logger.error(f"Error during /plot: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error during plotting: {e}")

@app.post("/api/custom", response_model=AgentResponse)
async def run_custom(request: CustomInstructionRequest):
    """Runs a custom instruction through the agent."""
    try:
        response_msg = await workflow_logic.run_custom_instruction(
            instruction=request.instruction
        )
        return AgentResponse(message=response_msg)
    except Exception as e:
        logger.error(f"Error during /custom: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error running custom instruction: {e}")

@app.post("/api/autopilot", response_model=AutoPilotResponse)
async def run_autopilot(request: AutoPilotRequest):
    """Runs the automated multi-iteration ML workflow."""
    try:
        result_dict = await workflow_logic.run_auto_pilot(
            data_path=request.data_path,
            target_column=request.target_column,
            task_type=request.task_type,
            iterations=request.iterations,
            plot_save_dir=request.plot_save_dir or 'plots'
        )
        # Similar check and serving consideration for plot_path as in /plot
        plot_path = result_dict.get("plot_path")
        if plot_path and not os.path.exists(plot_path):
             logger.warning(f"Auto-Pilot plot path reported but not found: {plot_path}")
             plot_path = None

        return AutoPilotResponse(message=result_dict["message"], plot_path=plot_path)
    except Exception as e:
        logger.error(f"Error during /autopilot: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error during Auto-Pilot execution: {e}")


# --- Project Management Endpoints (Basic Placeholders) ---

@app.get("/api/projects", response_model=ProjectListResponse)
async def get_projects():
    """Lists available projects (placeholder)."""
    try:
        projects = workflow_logic.list_projects()
        return ProjectListResponse(projects=projects)
    except Exception as e:
        logger.error(f"Error listing projects: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error listing projects: {e}")

@app.post("/api/load_project", response_model=ConfigResponse) # Use ConfigResponse for simple success/fail
async def load_project_api(request: PathRequest): # Use PathRequest to send project name
    """Loads a project (placeholder)."""
    project_name = request.path # Reusing 'path' field for project name
    try:
        success = workflow_logic.load_project(project_name)
        if success:
            return ConfigResponse(success=True, message=f"Project '{project_name}' loaded.")
        else:
            raise HTTPException(status_code=404, detail=f"Project '{project_name}' not found or failed to load.")
    except Exception as e:
        logger.error(f"Error loading project '{project_name}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error loading project: {e}")

# Add endpoints for creating, saving projects if needed


# --- Static File Serving (Example for Plots) ---
# If not using file:// protocol (e.g., pure web app), serve plots statically.
# from fastapi.staticfiles import StaticFiles
# Mount a directory to serve plots from the '/plots' URL path
# Ensure the directory exists relative to where the server runs
# plot_dir = os.path.abspath(os.path.join(workflow_logic.current_working_directory, 'plots'))
# os.makedirs(plot_dir, exist_ok=True)
# app.mount("/plots", StaticFiles(directory=plot_dir), name="plots")
# In PlotResponse, you would then return "/plots/your_plot_filename.png" instead of the absolute path.


# --- Main Entry Point (for running with uvicorn) ---
# This part is usually not needed if run via __main__.py but useful for direct testing
# if __name__ == "__main__":
#     import uvicorn
#     # Load environment variables (e.g., for OPENAI_API_KEY)
#     from dotenv import load_dotenv
#     load_dotenv()
#     print("Starting API server on http://localhost:8000")
#     # Set a default working directory on startup if needed
#     # workflow_logic.set_working_directory(".")
#     # Configure a default agent on startup? Or require user to configure via API?
#     # try:
#     #     workflow_logic.configure_agent({"provider": "openai", "api_key": os.getenv("OPENAI_API_KEY")})
#     # except Exception as e:
#     #     print(f"WARN: Could not configure default agent on startup: {e}")
#     uvicorn.run(app, host="127.0.0.1", port=8000)

