# ml_copilot_agent/workflow_logic.py

import os
import asyncio
import logging
from typing import Optional, List, Dict, Any
from llama_index.core import Settings
from llama_index.agent.openai import OpenAIAgent
from llama_index.llms.openai import OpenAI as LlamaOpenAI # Rename to avoid conflict
from llama_index.llms.ollama import Ollama
from llama_index.tools.code_interpreter.base import CodeInterpreterToolSpec
import pandas as pd
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Global Agent ---
# This agent will be configured by the API based on user settings
# We initialize it with None and configure it via configure_agent
agent: Optional[OpenAIAgent] = None
code_tools = CodeInterpreterToolSpec().to_tool_list()

# --- Configuration ---
# These will be updated by API calls
current_working_directory = os.getcwd()
project_metadata: Dict[str, Any] = {} # Holds metadata for the current project

# --- Helper Functions ---

def set_working_directory(path: str) -> bool:
    """Sets the current working directory."""
    global current_working_directory
    try:
        os.chdir(path)
        current_working_directory = os.getcwd()
        logger.info(f"Working directory changed to: {current_working_directory}")
        # Ensure necessary subdirectories exist (optional, based on project structure)
        os.makedirs(os.path.join(current_working_directory, 'data'), exist_ok=True)
        os.makedirs(os.path.join(current_working_directory, 'models'), exist_ok=True)
        os.makedirs(os.path.join(current_working_directory, 'results'), exist_ok=True)
        os.makedirs(os.path.join(current_working_directory, 'plots'), exist_ok=True) # Add plots dir
        return True
    except FileNotFoundError:
        logger.error(f"Directory not found: {path}")
        return False
    except Exception as e:
        logger.error(f"Error changing directory to {path}: {e}")
        return False

def configure_agent(config: Dict[str, Any]):
    """Configures the global agent based on provided settings."""
    global agent
    logger.info(f"Configuring agent with settings: {config}")
    try:
        provider = config.get("provider", "openai")
        temperature = config.get("temperature", 0.1) # Make temperature configurable if needed

        if provider == "ollama":
            ollama_model = config.get("ollama_model", "llama3")
            ollama_endpoint = config.get("ollama_endpoint", "http://localhost:11434")
            logger.info(f"Using Ollama LLM: model={ollama_model}, endpoint={ollama_endpoint}")
            # Note: Ollama might not work directly with OpenAIAgent if function calling isn't supported well.
            # This might require using a different agent type (e.g., ReActAgent with Ollama)
            # For now, we attempt to configure Settings.llm which OpenAIAgent uses by default.
            # Consider adding error handling or checks for Ollama's capabilities.
            Settings.llm = Ollama(model=ollama_model, base_url=ollama_endpoint, temperature=temperature, request_timeout=120.0)
            # Re-initialize agent with the new LLM in Settings
            agent = OpenAIAgent.from_tools(code_tools, llm=Settings.llm, verbose=True)

        elif provider == "openai":
            api_key = config.get("api_key")
            api_model = config.get("api_model", "gpt-4o")
            if not api_key:
                logger.warning("OpenAI API key not provided. Agent may not function.")
                # Set to None or raise error depending on desired behavior
                Settings.llm = None
                agent = None
                raise ValueError("OpenAI API Key is required but not provided.")

            logger.info(f"Using OpenAI LLM: model={api_model}")
            # Set the key in the environment for LlamaIndex OpenAI client
            os.environ["OPENAI_API_KEY"] = api_key
            Settings.llm = LlamaOpenAI(model=api_model, temperature=temperature)
            # Re-initialize agent with the new LLM in Settings
            agent = OpenAIAgent.from_tools(code_tools, llm=Settings.llm, verbose=True)

        else:
            logger.error(f"Unsupported LLM provider: {provider}")
            Settings.llm = None
            agent = None
            raise ValueError(f"Unsupported LLM provider: {provider}")

        logger.info("Agent configured successfully.")

    except Exception as e:
        logger.error(f"Failed to configure agent: {e}", exc_info=True)
        Settings.llm = None
        agent = None
        raise # Re-raise the exception to be caught by the API endpoint

async def run_agent_task(prompt: str) -> str:
    """Runs a prompt through the configured agent."""
    if agent is None or Settings.llm is None:
        logger.error("Agent is not configured. Please configure LLM settings first.")
        return "Error: Agent not configured. Check LLM settings."

    logger.info(f"Running agent task with prompt:\n{prompt[:200]}...") # Log truncated prompt
    try:
        # Use asyncio.to_thread for synchronous agent methods if needed,
        # but agent.chat might be async-compatible or handle its own threading.
        # If agent.chat is synchronous:
        # response = await asyncio.to_thread(agent.chat, prompt)

        # Assuming agent.chat can be awaited or handles its own execution context
        # If OpenAIAgent.chat is sync, use run_in_executor as in original workflow
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, agent.chat, prompt)

        logger.info(f"Agent response received: {response.response[:200]}...")
        return str(response) # Return the string representation of the agent's response
    except Exception as e:
        logger.error(f"Error during agent task execution: {e}", exc_info=True)
        return f"Error during agent execution: {e}"

# --- Core Task Functions ---

async def list_files_in_path(path: str) -> List[str]:
    """Lists files and directories in the specified path relative to CWD."""
    target_path = os.path.abspath(os.path.join(current_working_directory, path))
    logger.info(f"Listing files in: {target_path}")
    if not os.path.exists(target_path) or not os.path.isdir(target_path):
        logger.warning(f"Path not found or not a directory: {target_path}")
        return []
    try:
        items = []
        for item in os.listdir(target_path):
            full_item_path = os.path.join(target_path, item)
            if os.path.isdir(full_item_path):
                items.append(f"{item}/") # Add trailing slash for directories
            else:
                items.append(item)
        return items
    except Exception as e:
        logger.error(f"Error listing files in {target_path}: {e}")
        return []

async def get_csv_columns(file_path: str) -> List[str]:
    """Reads the header of a CSV file to get column names."""
    full_path = os.path.abspath(os.path.join(current_working_directory, file_path))
    logger.info(f"Getting columns from: {full_path}")
    if not os.path.exists(full_path):
        logger.error(f"File not found: {full_path}")
        raise FileNotFoundError(f"File not found: {full_path}")
    try:
        # Read only the first few rows to get header quickly
        df = pd.read_csv(full_path, nrows=5)
        return df.columns.tolist()
    except Exception as e:
        logger.error(f"Error reading CSV columns from {full_path}: {e}")
        raise ValueError(f"Could not read columns from file: {e}")

async def run_preprocessing_step(
    dataset_path: str,
    target_column: str,
    save_path: str,
    columns_to_drop: Optional[List[str]] = None,
    additional_instructions: Optional[str] = None
) -> str:
    """Generates and runs preprocessing code via the agent."""
    logger.info(f"Running preprocessing: Data='{dataset_path}', Target='{target_column}', Save='{save_path}'")

    # Ensure save directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    drop_instruction = ""
    if columns_to_drop:
        drop_instruction = f"- Drop the following columns: {columns_to_drop}."

    prompt = f"""
Context: You are an AI assistant helping with machine learning tasks. The current working directory is '{current_working_directory}'. All file paths should be interpreted relative to this directory unless they are absolute paths.

Task: Write and execute Python code to preprocess data for a binary classification task.

Details:
- Load the dataset from '{dataset_path}' into a pandas DataFrame.
- The target variable is '{target_column}'. Ensure this column remains unchanged during feature preprocessing.
{drop_instruction}
- Perform standard preprocessing steps suitable for binary classification, including:
    - Handle missing values appropriately (e.g., imputation).
    - Encode categorical features (e.g., one-hot encoding).
    - Scale or normalize numerical features.
- Apply the following additional user instructions if provided: '{additional_instructions or 'None'}'.
- Save the fully preprocessed data (including the target column) to '{save_path}'.
- Print a confirmation message upon successful completion, including the path where the data was saved.
"""
    return await run_agent_task(prompt)

async def run_training_step(
    data_path: str,
    target_column: str, # Needed to identify target during training
    model_save_path: str,
    task_type: str = "classification", # "classification" or "regression"
    additional_instructions: Optional[str] = None
) -> str:
    """Generates and runs model training code via the agent."""
    logger.info(f"Running training: Data='{data_path}', Save='{model_save_path}', Type='{task_type}'")

    # Ensure save directory exists
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    task_specific_instructions = ""
    if task_type == "classification":
        task_specific_instructions = "Train a suitable binary classification model (e.g., Logistic Regression, SVM, Random Forest, Gradient Boosting)."
    elif task_type == "regression":
        task_specific_instructions = "Train a suitable regression model (e.g., Linear Regression, Ridge, Lasso, Random Forest Regressor, Gradient Boosting Regressor)."
    else:
        return f"Error: Invalid task type '{task_type}'. Must be 'classification' or 'regression'."

    prompt = f"""
Context: You are an AI assistant helping with machine learning tasks. The current working directory is '{current_working_directory}'. All file paths should be interpreted relative to this directory unless they are absolute paths.

Task: Write and execute Python code to train a {task_type} model.

Details:
- Load the preprocessed dataset from '{data_path}'.
- Identify the target variable column named '{target_column}'. Separate features (X) and target (y).
- Split the data into training and testing sets (e.g., 80% train, 20% test, use a fixed random_state for reproducibility if possible).
- {task_specific_instructions} Use the training set for training.
- Apply the following additional user instructions if provided: '{additional_instructions or 'None'}'.
- Save the trained model object to '{model_save_path}' (e.g., using pickle or joblib).
- Print a confirmation message upon successful completion, including the path where the model was saved.
"""
    return await run_agent_task(prompt)

async def run_evaluation_step(
    data_path: str,
    target_column: str, # Needed to identify target during evaluation
    model_path: str,
    evaluation_save_path: str,
    task_type: str = "classification", # "classification" or "regression"
    additional_instructions: Optional[str] = None
) -> Dict[str, Any]: # Return dict containing message and potentially structured results
    """Generates and runs model evaluation code via the agent."""
    logger.info(f"Running evaluation: Data='{data_path}', Model='{model_path}', Save='{evaluation_save_path}', Type='{task_type}'")

    # Ensure save directory exists
    os.makedirs(os.path.dirname(evaluation_save_path), exist_ok=True)

    metrics_instructions = ""
    if task_type == "classification":
        metrics_instructions = "Calculate and report standard classification metrics: accuracy, precision, recall, F1-score, and AUC (Area Under ROC Curve). Use the test set for evaluation."
    elif task_type == "regression":
        metrics_instructions = "Calculate and report standard regression metrics: Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R-squared (R²). Use the test set for evaluation."
    else:
         return {"message": f"Error: Invalid task type '{task_type}'.", "results": None}

    prompt = f"""
Context: You are an AI assistant helping with machine learning tasks. The current working directory is '{current_working_directory}'. All file paths should be interpreted relative to this directory unless they are absolute paths.

Task: Write and execute Python code to evaluate a trained {task_type} model.

Details:
- Load the preprocessed dataset from '{data_path}'.
- Load the trained model from '{model_path}'.
- Identify the target variable column named '{target_column}'. Separate features (X) and target (y).
- Use the same train/test split strategy as used during training (e.g., 80/20 split with the same random_state if possible) to get the test set, OR assume the data at '{data_path}' is the full dataset and needs splitting now. **Crucially, evaluate only on the test set.**
- {metrics_instructions}
- Apply the following additional user instructions if provided: '{additional_instructions or 'None'}'.
- Save the calculated metrics (as a dictionary or plain text) to '{evaluation_save_path}'.
- **Important:** In your final output, include a JSON block containing the calculated metrics. Format it like this: ```json\n{{"metric1": value1, "metric2": value2, ...}}\n```. Also print a confirmation message.
"""
    response_str = await run_agent_task(prompt)

    # Try to parse metrics from the response
    results = None
    try:
        json_block = response_str.split("```json")[1].split("```")[0]
        results = json.loads(json_block.strip())
        logger.info(f"Parsed evaluation results: {results}")
    except Exception as e:
        logger.warning(f"Could not parse JSON metrics from agent response: {e}. Response was: {response_str[:500]}")

    return {"message": response_str, "results": results}


async def run_plotting_step(
    plot_type: str, # 'results' or 'data'
    data_file_path: str,
    target_column: Optional[str] = None, # Needed for some data plots
    plot_save_dir: str = "plots", # Save plots in a dedicated subdir
    additional_instructions: Optional[str] = None
) -> Dict[str, Any]: # Return dict containing message and plot path(s)
    """Generates and runs plotting code via the agent."""
    logger.info(f"Running plotting: Type='{plot_type}', Data='{data_file_path}', SaveDir='{plot_save_dir}'")

    # Ensure save directory exists
    full_plot_save_dir = os.path.abspath(os.path.join(current_working_directory, plot_save_dir))
    os.makedirs(full_plot_save_dir, exist_ok=True)

    # Generate a unique filename prefix for this plot run
    plot_filename_prefix = f"plot_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
    plot_save_path_template = os.path.join(full_plot_save_dir, f"{plot_filename_prefix}_[PLOT_TYPE].png") # Agent should replace [PLOT_TYPE]

    load_instruction = f"- Load the data from '{data_file_path}'."
    if plot_type == 'results':
        load_instruction = f"- Load the evaluation results (likely text or JSON) from '{data_file_path}'."

    default_plot_instruction = ""
    if plot_type == 'results':
        default_plot_instruction = "Create appropriate plots to visualize the evaluation metrics (e.g., bar chart of metrics)."
    elif plot_type == 'data':
        default_plot_instruction = f"Create exploratory data analysis plots (e.g., histograms for numerical features, bar charts for categorical features, correlation matrix, pair plots). Use target column '{target_column}' if relevant for coloring or grouping."

    prompt = f"""
Context: You are an AI assistant helping with machine learning tasks. The current working directory is '{current_working_directory}'. All file paths should be interpreted relative to this directory unless they are absolute paths.

Task: Write and execute Python code to generate and save plots for '{plot_type}'.

Details:
{load_instruction}
- Apply the following plotting instructions: '{additional_instructions if additional_instructions else default_plot_instruction}'.
- Use libraries like Matplotlib or Seaborn for plotting.
- Save the generated plot(s) as PNG files in the '{full_plot_save_dir}' directory. Use descriptive filenames starting with the prefix '{plot_filename_prefix}_' followed by the plot type (e.g., '{plot_filename_prefix}_accuracy_bar.png', '{plot_filename_prefix}_pca.png').
- **Important:** After saving the plot(s), print the full, absolute path(s) to the saved PNG file(s). Each path should be on a new line, enclosed in triple backticks, like this: ```\n/path/to/your/plot/plot_20250403_235500_accuracy_bar.png\n```. Also print a confirmation message.
"""
    response_str = await run_agent_task(prompt)

    # Try to parse plot paths from the response
    plot_paths = []
    try:
        lines = response_str.split('\n')
        for line in lines:
            if line.startswith('```') and line.endswith('```'):
                path = line.strip('`').strip()
                if path.lower().endswith('.png') and os.path.isabs(path): # Basic check
                     # Security check: Ensure path is within the intended plot directory
                     if os.path.commonpath([full_plot_save_dir, path]) == full_plot_save_dir:
                         plot_paths.append(path)
                     else:
                         logger.warning(f"Agent tried to report a plot path outside the allowed directory: {path}")

        logger.info(f"Parsed plot paths: {plot_paths}")
    except Exception as e:
        logger.warning(f"Could not parse plot paths from agent response: {e}. Response was: {response_str[:500]}")

    # For simplicity, return the first plot path found, or None
    # A real implementation might return all paths and let the frontend choose/display
    first_plot_path = plot_paths[0] if plot_paths else None

    return {"message": response_str, "plot_path": first_plot_path}

async def run_custom_instruction(instruction: str) -> str:
    """Generates and runs custom code via the agent."""
    logger.info(f"Running custom instruction: {instruction[:100]}...")
    prompt = f"""
Context: You are an AI assistant helping with machine learning tasks. The current working directory is '{current_working_directory}'. All file paths should be interpreted relative to this directory unless they are absolute paths.

Task: Execute the following custom instruction provided by the user. Write and execute the necessary Python code.

User Instruction:
{instruction}

Output the results or a confirmation message.
"""
    return await run_agent_task(prompt)


async def run_auto_pilot(
    data_path: str,
    target_column: str,
    task_type: str, # classification or regression
    iterations: int = 1,
    plot_save_dir: str = "plots"
) -> Dict[str, Any]:
    """Runs an end-to-end ML workflow multiple times with bootstrapping."""
    logger.info(f"Running Auto-Pilot: Data='{data_path}', Target='{target_column}', Type='{task_type}', Iterations={iterations}")

    # Ensure save directory exists
    full_plot_save_dir = os.path.abspath(os.path.join(current_working_directory, plot_save_dir))
    os.makedirs(full_plot_save_dir, exist_ok=True)
    results_save_path = os.path.join(current_working_directory, 'results', f"autopilot_metrics_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json")
    plot_save_path = os.path.join(full_plot_save_dir, f"autopilot_boxplot_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.png")

    metrics_list_str = "accuracy, precision, recall, f1, auc" if task_type == "classification" else "mae, mse, rmse, r2"

    prompt = f"""
Context: You are an AI assistant executing an automated end-to-end machine learning workflow. The current working directory is '{current_working_directory}'. All file paths should be interpreted relative to this directory unless they are absolute paths.

Task: Perform an automated ML workflow {iterations} times for a {task_type} task.

Details:
1. Load the dataset from '{data_path}'. Target column is '{target_column}'.
2. Inside a loop running {iterations} times:
    a. Preprocess the data (handle missing values, encode categoricals, scale numericals). **Important:** Perform preprocessing freshly inside the loop if it involves data splitting or fitting (like scalers) to avoid data leakage across iterations. If preprocessing is deterministic and doesn't involve fitting (e.g., simple value replacement), it can be done once before the loop. Use your judgment.
    b. Split the (potentially freshly preprocessed) data into train and test sets using a **different random seed** for each iteration to simulate bootstrapping.
    c. Train a simple baseline {task_type} model (e.g., Logistic Regression/Linear Regression or RandomForest) on the training set.
    d. Evaluate the model on the test set, calculating metrics: {metrics_list_str}.
    e. Store the metrics for this iteration (e.g., in a list of dictionaries).
3. After the loop finishes:
    a. Save all collected metrics from all iterations into a JSON file at '{results_save_path}'.
    b. Create a box plot visualizing the distribution of each metric across the {iterations} runs. Use Seaborn or Matplotlib. Label the axes clearly (Metrics on X-axis, Score on Y-axis).
    c. Save the box plot as a PNG file to '{plot_save_path}'.
4. **Important Output:**
    a. Print a message confirming the completion of all iterations.
    b. Print the absolute path to the saved JSON metrics file, enclosed in triple backticks: ```\n{results_save_path}\n```
    c. Print the absolute path to the saved box plot PNG file, enclosed in triple backticks: ```\n{plot_save_path}\n```
"""
    response_str = await run_agent_task(prompt)

    # Try to parse the plot path
    final_plot_path = None
    try:
        lines = response_str.split('\n')
        for line in lines:
            if line.startswith('```') and line.endswith('```'):
                path = line.strip('`').strip()
                if path.lower().endswith('.png') and os.path.isabs(path) and "autopilot_boxplot" in os.path.basename(path):
                     if os.path.commonpath([full_plot_save_dir, path]) == full_plot_save_dir:
                         final_plot_path = path
                         break # Assume the first matching path is the one we want
                     else:
                          logger.warning(f"Auto-Pilot: Agent reported plot path outside allowed directory: {path}")

        logger.info(f"Auto-Pilot parsed plot path: {final_plot_path}")
    except Exception as e:
        logger.warning(f"Auto-Pilot: Could not parse plot path from agent response: {e}. Response was: {response_str[:500]}")

    return {"message": response_str, "plot_path": final_plot_path}

# --- Project Management (Basic Placeholder) ---

def load_project(project_name: str) -> bool:
    """Loads project metadata (placeholder)."""
    global project_metadata
    # In a real app, load from a file like projects/{project_name}/metadata.json
    logger.info(f"Loading project: {project_name}")
    # Simulate loading - replace with actual file I/O
    project_metadata = {"name": project_name, "last_accessed": str(pd.Timestamp.now())}
    # Maybe set working directory based on project?
    # set_working_directory(os.path.join(base_project_dir, project_name))
    return True

def save_project() -> bool:
    """Saves project metadata (placeholder)."""
    if not project_metadata:
        logger.warning("No project loaded, cannot save.")
        return False
    project_name = project_metadata.get("name")
    if not project_name:
        logger.error("Project metadata is missing name, cannot save.")
        return False
    logger.info(f"Saving project: {project_name}")
    # In a real app, save project_metadata to projects/{project_name}/metadata.json
    return True

def list_projects() -> List[str]:
    """Lists available projects (placeholder)."""
    logger.info("Listing projects")
    # In a real app, list directories in a base 'projects' folder
    return ["project_alpha", "example_project"]

