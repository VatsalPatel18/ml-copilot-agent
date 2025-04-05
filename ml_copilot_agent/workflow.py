import os
import asyncio
import json # Added for potentially handling complex parameters
from typing import Optional, Union, List, Dict, Any # Added List, Dict, Any

from llama_index.core.workflow import Workflow, Context, Event, StartEvent, StopEvent, step
from llama_index.tools.code_interpreter.base import CodeInterpreterToolSpec
# Assuming OpenAI Agent setup remains similar
from llama_index.agent.openai import OpenAIAgent
from llama_index.core import Settings
# from llama_index.utils.workflow import draw_all_possible_flows # Keep if needed for visualization

# ===========================================================
# 1. Enhanced Event Definitions
# ===========================================================

# --- Core Events ---
class InitializeEvent(Event):
    """Event triggered at the start of the workflow."""
    pass

class ML_Copilot_Command(Event):
    """Event carrying the user's high-level command."""
    user_command: str

class StopWorkflowEvent(StopEvent):
    """Event to signal stopping the workflow."""
    result: str = "Workflow stopped."

# --- Data Handling Events ---
class LoadDataEvent(Event):
    """Event to load one or more datasets."""
    datasets_to_load: List[Dict[str, str]] # List of dicts, each with 'path', 'index_col', 'var_name', 'required_cols' (optional string list)

class PreprocessEvent(Event):
    """Event for data preprocessing tasks."""
    input_vars: List[str] # Variable names of datasets to preprocess
    align_indices: bool = False
    steps: List[Dict[str, Any]] # List of dicts, e.g., {'type': 'handle_missing', 'method': 'mean'}, {'type': 'scale', 'method': 'MinMaxScaler'}
    subset_based_on_labels: Optional[Dict[str, Any]] = None # e.g., {'label_var': 'clusters', 'labels_to_keep': [0, 1]}
    output_vars: Dict[str, str] # e.g., {'features': 'X_processed', 'labels': 'y_processed'}
    save_path: Optional[str] = None
    additional_instructions: Optional[str] = ""

# --- ML Task Events ---
class ClusterEvent(Event):
    """Event for clustering tasks."""
    input_var: str # Variable name of dataset to cluster
    methods: List[str] # e.g., ['kmeans', 'agglomerative']
    k_range: List[int] # e.g., list(range(2, 11))
    evaluate_survival: bool = False
    survival_data_var: Optional[str] = None
    time_col: Optional[str] = None
    event_col: Optional[str] = None
    output_best_result_var: str # Variable name for dict containing best labels, model, info
    save_path_prefix: Optional[str] = None # Prefix for saving plots/results
    additional_instructions: Optional[str] = ""

class TrainEvent(Event):
    """Event for training models."""
    task_type: str # 'binary classification', 'regression', etc.
    features_var: str
    target_var: str
    models: List[Dict[str, Any]] # List of dicts, e.g., {'name': 'LogisticRegression', 'params': {'max_iter': 2000}}
    feature_selection: Optional[Dict[str, Any]] = None # e.g., {'method': 'SelectKBest-f_classif', 'k_values': [10, 50, 100]}
    cross_validation: Optional[Dict[str, Any]] = None # e.g., {'type': 'runs', 'n': 10} or {'type': 'kfold', 'n': 5}
    save_path_prefix: str # Base path/prefix for models, features, metrics
    output_metrics_var: str # Variable name for detailed metrics DataFrame
    output_roc_data_var: Optional[str] = None # Variable name for ROC data storage
    additional_instructions: Optional[str] = ""

class EvaluateEvent(Event):
    """Event for evaluating models."""
    models_source: Union[str, List[str]] # Variable name of results df, path pattern, or list of model paths
    datasets: List[Dict[str, str]] # List of dicts, each with 'name', 'features_var', 'labels_var', 'survival_var' (optional)
    metrics: List[str] # e.g., ['AUC', 'Accuracy', 'LogRankPValue']
    time_col: Optional[str] = None # Required if LogRankPValue is requested
    event_col: Optional[str] = None # Required if LogRankPValue is requested
    output_summary_var: str # Variable name for the aggregated evaluation summary
    save_path_prefix: Optional[str] = None # Prefix for saving detailed results/predictions
    additional_instructions: Optional[str] = ""

class SelectModelEvent(Event):
    """Event to select the best model based on criteria."""
    summary_table_var: str # Variable name of the evaluation summary table
    criteria: List[Dict[str, Any]] # List of dicts defining selection strategies, e.g., {'name': 'Strict', 'conditions': ['TCGA_p < 0.05', 'CPTAC_p < 0.05'], 'optimize': 'maximize AUC'}
    output_best_config_var: str # Variable name for the selected model's config
    models_base_path: str # Path needed to locate the actual model file later
    additional_instructions: Optional[str] = ""

# --- Analysis & Utility Events ---
class AnalyzeSurvivalEvent(Event):
    """Event for Kaplan-Meier analysis."""
    input_df_var: str # DataFrame containing survival data and group labels
    time_col: str
    event_col: str
    group_col: str
    compare_groups: Union[str, List[List[Any]]] # e.g., 'all pairwise', [[0, 1]]
    plot_title: str
    plot_save_path: str
    results_save_path: Optional[str] = None # Path to save log-rank results
    additional_instructions: Optional[str] = ""

class PlotEvent(Event):
    """Event for generating various plots."""
    plot_type: str # 'Kaplan-Meier', 'ROC Curve', 'AUC Boxplot', etc.
    input_data: Dict[str, str] # Dict mapping required input names to variable names/paths, e.g., {'metrics_df': 'df_metrics', 'roc_data': 'roc_storage'}
    parameters: Dict[str, Any] # e.g., {'title': 'My Plot', 'highlight': 'best_model'}
    save_path: str
    additional_instructions: Optional[str] = ""

class ListFilesEvent(Event):
    """Event to list files in the current directory."""
    pass

class CustomTaskEvent(Event):
    """Event for custom, user-defined tasks."""
    instruction: str
    input_vars: Optional[Dict[str, str]] = None # Map input description to variable name
    output_description: Optional[str] = None # Describe expected output

# ===========================================================
# 2. Enhanced MLWorkflow Definition
# ===========================================================
class MLWorkflow(Workflow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize the code interpreter tool (ensure API key is set elsewhere)
        code_spec = CodeInterpreterToolSpec()
        tools = code_spec.to_tool_list()
        # Create the OpenAIAgent with the code interpreter tool
        # Ensure Settings.llm is configured appropriately before instantiation
        if not Settings.llm:
             raise ValueError("LLM not configured in Settings. Please set Settings.llm.")
        self.agent = OpenAIAgent.from_tools(tools, llm=Settings.llm, verbose=True)
        self.llm = Settings.llm # Keep reference if needed directly

    # Helper for structured input
    def _get_input(self, prompt: str, required: bool = True, default: Optional[str] = None) -> Optional[str]:
        """Helper to get user input with required/default handling."""
        while True:
            default_str = f" (default: {default})" if default else ""
            required_str = " (required)" if required else ""
            full_prompt = f"{prompt}{required_str}{default_str}: "
            value = input(full_prompt).strip()
            if value:
                return value
            elif default is not None:
                return default
            elif required:
                print("This input is required.")
            else: # Not required, no default, no input -> return None
                return None

    def _get_list_input(self, prompt: str, separator: str = ',') -> List[str]:
        """Helper to get comma-separated list input."""
        while True:
            value = self._get_input(f"{prompt} (comma-separated)", required=True)
            if value:
                items = [item.strip() for item in value.split(separator) if item.strip()]
                if items:
                    return items
                else:
                    print("Please enter at least one item.")
            # This loop should not exit without valid input due to required=True in _get_input

    def _get_dict_input(self, prompt: str) -> Dict[str, Any]:
        """Helper to get simple key-value pairs (e.g., for parameters)."""
        print(f"{prompt} (Enter key=value pairs, one per line. Press Enter on empty line to finish):")
        data = {}
        while True:
            line = input("> ").strip()
            if not line:
                break
            if '=' in line:
                key, value = line.split('=', 1)
                # Attempt to parse value (basic types)
                try:
                    data[key.strip()] = json.loads(value.strip())
                except json.JSONDecodeError:
                    data[key.strip()] = value.strip() # Store as string if not valid JSON
            else:
                print("Invalid format. Please use key=value.")
        return data

    def _get_bool_input(self, prompt: str, default: Optional[bool] = None) -> bool:
        """Helper to get boolean input."""
        default_val = None
        if default is True: default_val = 'yes'
        if default is False: default_val = 'no'

        while True:
            response = self._get_input(f"{prompt} (yes/no)", required=default is None, default=default_val).lower()
            if response in ['yes', 'y']:
                return True
            elif response in ['no', 'n']:
                return False
            else:
                print("Please enter 'yes' or 'no'.")


    # --- Workflow Steps ---

    @step
    async def initialize(self, ctx: Context, ev: StartEvent) -> ML_Copilot_Command:
        """Initializes the workflow and gets the first user command."""
        print("--- ML Copilot Initialized ---")
        print("Enter your command (e.g., 'load data', 'preprocess', 'cluster', 'train', 'evaluate', 'plot', 'analyze survival', 'select model', 'list files', 'custom task', 'help', 'exit').")
        user_command = self._get_input("> ", required=True)
        return ML_Copilot_Command(user_command=user_command)

    @step
    async def ml_copilot_router(self, ctx: Context, ev: ML_Copilot_Command) -> LoadDataEvent| PreprocessEvent| ClusterEvent| TrainEvent| EvaluateEvent| SelectModelEvent| AnalyzeSurvivalEvent| PlotEvent| ListFilesEvent| CustomTaskEvent | StopWorkflowEvent | ML_Copilot_Command | StopEvent:
        """Routes the user command to the appropriate event creation logic."""
        command = ev.user_command.lower().strip()

        # --- Routing Logic ---
        if "load data" in command:
            print("--- Loading Data ---")
            datasets = []
            while True:
                path = self._get_input("Dataset file path")
                index_col = self._get_input("Index column (e.g., PatientID)")
                var_name = self._get_input("Variable name for this dataset (e.g., df_expr)")
                required_cols_str = self._get_input("Required columns (comma-separated, optional)", required=False)
                required_cols = [c.strip() for c in required_cols_str.split(',')] if required_cols_str else None
                datasets.append({'path': path, 'index_col': index_col, 'var_name': var_name, 'required_cols': required_cols})
                if not self._get_bool_input("Load another dataset?", default=False):
                    break
            return LoadDataEvent(datasets_to_load=datasets)

        elif "preprocess" in command:
            print("--- Preprocessing Data ---")
            input_vars = self._get_list_input("Input dataset variable names")
            align_indices = self._get_bool_input("Align datasets on common indices?", default=False)
            print("Specify preprocessing steps (e.g., {'type': 'handle_missing', 'method': 'mean'}). Enter one JSON-like step per line, empty line to finish:")
            steps = []
            while True:
                line = input("> ").strip()
                if not line: break
                try: steps.append(json.loads(line))
                except json.JSONDecodeError: print("Invalid JSON format. Please try again.")
            subset_labels = None
            if self._get_bool_input("Subset data based on prior labels/clusters?", default=False):
                label_var = self._get_input("Variable name of DataFrame with labels")
                labels_to_keep_str = self._get_input("Labels/groups to keep (comma-separated or JSON list)")
                try: labels_to_keep = json.loads(labels_to_keep_str)
                except: labels_to_keep = [l.strip() for l in labels_to_keep_str.split(',')]
                subset_labels = {'label_var': label_var, 'labels_to_keep': labels_to_keep}
            print("Specify output variable names (e.g., features=X_processed, labels=y_processed). Enter key=value pairs:")
            output_vars = self._get_dict_input("Output variable names")
            save_path = self._get_input("Save preprocessed data path (optional)", required=False)
            additional_instructions = self._get_input("Any additional instructions (optional)", required=False)
            return PreprocessEvent(input_vars=input_vars, align_indices=align_indices, steps=steps, subset_based_on_labels=subset_labels, output_vars=output_vars, save_path=save_path, additional_instructions=additional_instructions)

        elif "cluster" in command:
            print("--- Clustering Data ---")
            input_var = self._get_input("Input dataset variable name for clustering")
            methods = self._get_list_input("Clustering methods to try (e.g., kmeans, gmm)")
            k_range_str = self._get_input("Range of k values (e.g., 2-10 or JSON list)")
            try: k_range = json.loads(k_range_str)
            except: k_start, k_end = map(int, k_range_str.split('-')); k_range = list(range(k_start, k_end + 1))
            evaluate_survival = self._get_bool_input("Evaluate survival separation?", default=False)
            survival_data_var, time_col, event_col = None, None, None
            if evaluate_survival:
                survival_data_var = self._get_input("Survival data variable name")
                time_col = self._get_input("Time column name")
                event_col = self._get_input("Event column name")
            output_best_result_var = self._get_input("Output variable name for best clustering result info")
            save_path_prefix = self._get_input("Path prefix for saving plots/results (optional)", required=False)
            additional_instructions = self._get_input("Any additional instructions (optional)", required=False)
            return ClusterEvent(input_var=input_var, methods=methods, k_range=k_range, evaluate_survival=evaluate_survival, survival_data_var=survival_data_var, time_col=time_col, event_col=event_col, output_best_result_var=output_best_result_var, save_path_prefix=save_path_prefix, additional_instructions=additional_instructions)

        elif "train" in command:
            print("--- Training Model(s) ---")
            task_type = self._get_input("Task type (e.g., binary classification)")
            features_var = self._get_input("Input features variable name")
            target_var = self._get_input("Target variable name")
            print("Specify models to train (e.g., {'name': 'LogisticRegression', 'params': {'max_iter': 2000}}). Enter one JSON-like model spec per line, empty line to finish:")
            models = []
            while True:
                line = input("> ").strip()
                if not line: break
                try: models.append(json.loads(line))
                except json.JSONDecodeError: print("Invalid JSON format. Please try again.")
            feature_selection = None
            if self._get_bool_input("Perform feature selection?", default=False):
                fs_method = self._get_input("Feature selection method (e.g., SelectKBest-f_classif)")
                k_values_str = self._get_input("Number/range of features (e.g., 10 or [10, 50, 100])")
                try: k_values = json.loads(k_values_str)
                except: k_values = int(k_values_str) # Assume single int if not list
                feature_selection = {'method': fs_method, 'k_values': k_values}
            cross_validation = None
            if self._get_bool_input("Run multiple train/test splits or cross-validation?", default=False):
                cv_type = self._get_input("Type ('runs' or 'kfold')")
                cv_n = int(self._get_input("Number of runs/folds"))
                cross_validation = {'type': cv_type, 'n': cv_n}
            save_path_prefix = self._get_input("Base path/prefix for saving models, features, metrics")
            output_metrics_var = self._get_input("Variable name for detailed metrics DataFrame")
            output_roc_data_var = self._get_input("Variable name for ROC data storage (optional, for plotting)", required=False)
            additional_instructions = self._get_input("Any additional instructions (optional)", required=False)
            return TrainEvent(task_type=task_type, features_var=features_var, target_var=target_var, models=models, feature_selection=feature_selection, cross_validation=cross_validation, save_path_prefix=save_path_prefix, output_metrics_var=output_metrics_var, output_roc_data_var=output_roc_data_var, additional_instructions=additional_instructions)

        elif "evaluate" in command:
            print("--- Evaluating Model(s) ---")
            models_source = self._get_input("Source of models/results (variable name or path pattern)")
            print("Specify validation datasets. Enter one dataset spec per line (JSON like {'name': 'TCGA', 'features_var': 'X_tcga', ...}), empty line to finish:")
            datasets = []
            while True:
                line = input("> ").strip()
                if not line: break
                try: datasets.append(json.loads(line))
                except json.JSONDecodeError: print("Invalid JSON format. Please try again.")
            metrics = self._get_list_input("Metrics to calculate (e.g., AUC, Accuracy, LogRankPValue)")
            time_col, event_col = None, None
            if 'LogRankPValue' in metrics:
                time_col = self._get_input("Time column name (for LogRankPValue)")
                event_col = self._get_input("Event column name (for LogRankPValue)")
            output_summary_var = self._get_input("Variable name for the aggregated evaluation summary")
            save_path_prefix = self._get_input("Path prefix for saving detailed results/predictions (optional)", required=False)
            additional_instructions = self._get_input("Any additional instructions (optional)", required=False)
            return EvaluateEvent(models_source=models_source, datasets=datasets, metrics=metrics, time_col=time_col, event_col=event_col, output_summary_var=output_summary_var, save_path_prefix=save_path_prefix, additional_instructions=additional_instructions)

        elif "select model" in command:
            print("--- Selecting Best Model ---")
            summary_table_var = self._get_input("Variable name of the evaluation summary table")
            print("Specify selection criteria strategies (e.g., {'name': 'Strict', 'conditions': ['TCGA_p < 0.05'], 'optimize': 'maximize AUC'}). Enter one JSON-like strategy per line, empty line to finish:")
            criteria = []
            while True:
                line = input("> ").strip()
                if not line: break
                try: criteria.append(json.loads(line))
                except json.JSONDecodeError: print("Invalid JSON format. Please try again.")
            output_best_config_var = self._get_input("Variable name for the selected model's config")
            models_base_path = self._get_input("Base path where models are saved (needed to find the selected model file)")
            additional_instructions = self._get_input("Any additional instructions (optional)", required=False)
            return SelectModelEvent(summary_table_var=summary_table_var, criteria=criteria, output_best_config_var=output_best_config_var, models_base_path=models_base_path, additional_instructions=additional_instructions)

        elif "analyze survival" in command:
            print("--- Analyzing Survival ---")
            input_df_var = self._get_input("Input DataFrame variable name (with survival data and groups)")
            time_col = self._get_input("Time column name")
            event_col = self._get_input("Event column name")
            group_col = self._get_input("Group label column name")
            compare_groups_str = self._get_input("Groups to compare ('all pairwise' or JSON list of lists e.g., [[0, 1]])")
            try: compare_groups = json.loads(compare_groups_str)
            except: compare_groups = compare_groups_str # Assume 'all pairwise' or similar string
            plot_title = self._get_input("Plot title")
            plot_save_path = self._get_input("Path to save KM plot")
            results_save_path = self._get_input("Path to save log-rank results (optional)", required=False)
            additional_instructions = self._get_input("Any additional instructions (optional)", required=False)
            return AnalyzeSurvivalEvent(input_df_var=input_df_var, time_col=time_col, event_col=event_col, group_col=group_col, compare_groups=compare_groups, plot_title=plot_title, plot_save_path=plot_save_path, results_save_path=results_save_path, additional_instructions=additional_instructions)

        elif "plot" in command or "visualize" in command:
            print("--- Plotting ---")
            plot_type = self._get_input("Type of plot (e.g., Kaplan-Meier, ROC Curve, AUC Boxplot)")
            print("Specify input data needed (e.g., metrics_df=df_metrics, roc_data=roc_storage). Enter key=value pairs:")
            input_data = self._get_dict_input("Input data mapping")
            print("Specify plot parameters (e.g., title='My Plot', highlight='best'). Enter key=value pairs:")
            parameters = self._get_dict_input("Plot parameters")
            save_path = self._get_input("Path to save the plot")
            additional_instructions = self._get_input("Any additional instructions (optional)", required=False)
            return PlotEvent(plot_type=plot_type, input_data=input_data, parameters=parameters, save_path=save_path, additional_instructions=additional_instructions)

        elif "list files" in command or "show files" in command:
            return ListFilesEvent()

        elif "custom" in command or "instruction" in command:
             print("--- Custom Task ---")
             instruction = self._get_input("Describe the custom task in detail")
             print("Specify input variables needed (e.g., input_df=df_data). Enter key=value pairs (optional):")
             input_vars = self._get_dict_input("Input variables")
             output_description = self._get_input("Describe the expected output (optional)", required=False)
             return CustomTaskEvent(instruction=instruction, input_vars=input_vars or None, output_description=output_description)

        elif "help" in command:
            print("\nAvailable Commands:")
            print("- load data: Load datasets into memory.")
            print("- preprocess: Clean, transform, and prepare data.")
            print("- cluster: Perform unsupervised clustering.")
            print("- train: Train machine learning models.")
            print("- evaluate: Evaluate model performance.")
            print("- select model: Choose the best model based on criteria.")
            print("- analyze survival: Perform Kaplan-Meier analysis.")
            print("- plot: Generate various plots.")
            print("- list files: Show files in the current directory.")
            print("- custom task: Execute a custom instruction.")
            print("- help: Show this help message.")
            print("- exit: Terminate the workflow.\n")
            # Get next command after showing help
            next_command = self._get_input("> ", required=True)
            return ML_Copilot_Command(user_command=next_command)

        elif "exit" in command or "quit" in command:
            return StopWorkflowEvent(result="Workflow terminated by user.")

        else:
            print(f"Command '{command}' not recognized. Type 'help' for available commands.")
            next_command = self._get_input("> ", required=True)
            return ML_Copilot_Command(user_command=next_command) # Re-prompt

    async def _execute_code_prompt(self, ctx: Context, prompt: str, step_name: str):
        """Helper to execute a prompt using the agent and print response."""
        print(f"\n--- Executing Step: {step_name} ---")
        print("Sending prompt to LLM for code generation and execution...")
        # print(f"Prompt:\n```\n{prompt}\n```") # Optional: print the prompt for debugging
        try:
            # Use the agent to generate and execute code asynchronously
            response = await asyncio.get_event_loop().run_in_executor(
                None, self.agent.chat, prompt
            )
            # Print the agent's response (code execution output)
            print("\n--- Agent Response ---")
            print(response)
            print("--------------------\n")
            # Check for errors indicated in the response (basic check)
            if "error" in str(response).lower() or "exception" in str(response).lower():
                 print(f"Warning: Potential error detected in {step_name} execution.")
                 # Optionally, store error state in context: await ctx.set(f'{step_name}_error', True)
            # else:
                 # Optionally, confirm success: await ctx.set(f'{step_name}_error', False)

        except Exception as e:
            print(f"\n--- Error during {step_name} execution ---")
            print(f"An exception occurred: {e}")
            print("-----------------------------------------\n")
            # Store error state in context
            # await ctx.set(f'{step_name}_error', True)
            # Note: Depending on workflow design, you might want to transition to an error state or stop

    # --- Step Implementations ---

    @step
    async def load_data(self, ctx: Context, ev: LoadDataEvent) -> ML_Copilot_Command:
        """Loads datasets based on the LoadDataEvent."""
        prompt_lines = ["Please write and execute Python code to perform the following data loading tasks:"]
        loaded_vars = []
        for ds in ev.datasets_to_load:
            path = ds['path']
            index_col = ds['index_col']
            var_name = ds['var_name']
            required_cols = ds.get('required_cols') # Optional

            load_instruction = f"- Load dataset from '{path}' into a pandas DataFrame named `{var_name}`."
            if index_col:
                load_instruction += f" Use column '{index_col}' as the index."
            prompt_lines.append(load_instruction)

            # Add check for required columns if specified
            if required_cols:
                 cols_str = "', '".join(required_cols)
                 prompt_lines.append(f"- Verify that `{var_name}` contains the required columns: ['{cols_str}']. Raise an error if any are missing.")

            # Basic check after loading
            prompt_lines.append(f"- Print the shape and first 5 rows of `{var_name}`.")
            loaded_vars.append(var_name)

        prompt = "\n".join(prompt_lines)
        await self._execute_code_prompt(ctx, prompt, "Load Data")
        await ctx.set('loaded_data_vars', loaded_vars) # Store names of loaded variables

        print(f"Data loading complete. Loaded variables: {', '.join(loaded_vars)}")
        print("What would you like to do next?")
        user_command = self._get_input("> ", required=True)
        return ML_Copilot_Command(user_command=user_command)

    @step
    async def preprocess(self, ctx: Context, ev: PreprocessEvent) -> ML_Copilot_Command:
        """Performs preprocessing based on the PreprocessEvent."""
        prompt_lines = ["Please write and execute Python code for data preprocessing:"]
        # Assume input variables are already loaded pandas DataFrames in the execution environment
        prompt_lines.append(f"# Input DataFrames: {', '.join(f'`{v}`' for v in ev.input_vars)}")

        if ev.align_indices:
            prompt_lines.append(f"- Align the following DataFrames based on their common indices: {', '.join(f'`{v}`' for v in ev.input_vars)}. Update the variables in place or assign to new ones if necessary, ensuring consistency.")

        prompt_lines.append(f"# Apply the following preprocessing steps sequentially:")
        for i, step_info in enumerate(ev.steps):
            step_type = step_info.get('type', 'unknown')
            # Add specific instructions based on step type
            if step_type == 'handle_missing':
                method = step_info.get('method', 'default_handling')
                cols = step_info.get('columns', 'all') # Target specific columns or all
                prompt_lines.append(f"- Step {i+1} ({step_type}): Handle missing values using method '{method}' for columns '{cols}'. Apply to relevant input DataFrames.")
            elif step_type == 'encode_categorical':
                method = step_info.get('method', 'one-hot')
                cols = step_info.get('columns', 'categorical_features') # Identify or specify columns
                prompt_lines.append(f"- Step {i+1} ({step_type}): Encode categorical features using method '{method}' for columns '{cols}'. Apply to relevant input DataFrames.")
            elif step_type == 'scale':
                method = step_info.get('method', 'StandardScaler')
                cols = step_info.get('columns', 'numerical_features') # Identify or specify columns
                exclude = step_info.get('exclude', []) # List of columns to exclude (e.g., target)
                prompt_lines.append(f"- Step {i+1} ({step_type}): Scale numerical features using '{method}' for columns '{cols}', excluding {exclude}. Apply to relevant input DataFrames.")
            elif step_type == 'subset_criteria':
                 criteria = step_info.get('criteria', 'no_criteria_provided')
                 prompt_lines.append(f"- Step {i+1} ({step_type}): Subset the relevant DataFrame based on criteria: {criteria}.")
            else:
                 prompt_lines.append(f"- Step {i+1} ({step_type}): Apply custom step: {json.dumps(step_info)}") # Generic fallback

        if ev.subset_based_on_labels:
            label_var = ev.subset_based_on_labels['label_var']
            labels_to_keep = ev.subset_based_on_labels['labels_to_keep']
            prompt_lines.append(f"- Subset the primary data DataFrame(s) to include only samples where the label in `{label_var}` is one of {labels_to_keep}.")
            prompt_lines.append(f"- Ensure all related DataFrames (features, labels, survival) are subsetted consistently based on the selected samples.")

        prompt_lines.append(f"# Assign the final preprocessed data to the following variables:")
        for key, var_name in ev.output_vars.items():
            prompt_lines.append(f"- Assign the {key} data to variable `{var_name}`.")
            prompt_lines.append(f"- Print the shape and type of `{var_name}`.") # Verify output

        if ev.save_path:
            # Be specific about which variable to save if multiple outputs
            var_to_save = list(ev.output_vars.values())[0] # Default to saving the first output var
            if len(ev.output_vars) > 1: print(f"Warning: Multiple output vars specified for saving. Saving '{var_to_save}' to '{ev.save_path}'. Adjust prompt if needed.")
            prompt_lines.append(f"- Save the final preprocessed data from variable `{var_to_save}` to '{ev.save_path}'.")

        if ev.additional_instructions:
            prompt_lines.append(f"\n# Additional User Instructions:\n{ev.additional_instructions}")

        prompt = "\n".join(prompt_lines)
        await self._execute_code_prompt(ctx, prompt, "Preprocess Data")
        # Store output variable names for potential future use
        await ctx.set('last_preprocessed_vars', ev.output_vars)

        print("Preprocessing complete.")
        print("What would you like to do next?")
        user_command = self._get_input("> ", required=True)
        return ML_Copilot_Command(user_command=user_command)


    @step
    async def cluster(self, ctx: Context, ev: ClusterEvent) -> ML_Copilot_Command:
        """Performs clustering based on the ClusterEvent."""
        prompt_lines = [f"Please write and execute Python code to perform clustering on the dataset in variable `{ev.input_var}`:"]

        prompt_lines.append(f"# Perform clustering using the following methods: {', '.join(ev.methods)}")
        prompt_lines.append(f"# Test the following numbers of clusters (k): {ev.k_range}")

        if ev.evaluate_survival:
            prompt_lines.append(f"\n# For each clustering result (method and k):")
            prompt_lines.append(f"- Evaluate survival separation using Kaplan-Meier analysis.")
            prompt_lines.append(f"- Use survival data from variable `{ev.survival_data_var}`.")
            prompt_lines.append(f"- The time column is '{ev.time_col}' and the event column is '{ev.event_col}'.")
            prompt_lines.append(f"- Calculate pairwise log-rank p-values between clusters.")
            if ev.save_path_prefix:
                prompt_lines.append(f"- Save the Kaplan-Meier plot for each result to a file starting with the prefix '{ev.save_path_prefix}' followed by method and k (e.g., '{ev.save_path_prefix}_kmeans_k3_km.png').")
            prompt_lines.append(f"- Keep track of the minimum pairwise p-value for each result.")

        prompt_lines.append(f"\n# Identify the best clustering configuration (method and k) based on the lowest minimum pairwise log-rank p-value (if survival evaluated) or other standard clustering metrics (like silhouette score) if survival not evaluated.")
        prompt_lines.append(f"- Store the results of the best configuration (cluster labels, model object, method name, k, best metric value) in a dictionary variable named `{ev.output_best_result_var}`.")
        prompt_lines.append(f"- Print the details of the best configuration found.")
        if ev.save_path_prefix and ev.evaluate_survival:
             prompt_lines.append(f"- Generate and save a final KM plot specifically for the best configuration using the prefix '{ev.save_path_prefix}_best'.")

        if ev.additional_instructions:
            prompt_lines.append(f"\n# Additional User Instructions:\n{ev.additional_instructions}")

        prompt = "\n".join(prompt_lines)
        await self._execute_code_prompt(ctx, prompt, "Clustering")
        await ctx.set('last_cluster_result_var', ev.output_best_result_var) # Store name of result var

        print("Clustering complete.")
        print("What would you like to do next?")
        user_command = self._get_input("> ", required=True)
        return ML_Copilot_Command(user_command=user_command)

    @step
    async def train(self, ctx: Context, ev: TrainEvent) -> ML_Copilot_Command:
        """Trains models based on the TrainEvent."""
        prompt_lines = [f"Please write and execute Python code for a '{ev.task_type}' task:"]
        prompt_lines.append(f"# Use features from variable `{ev.features_var}` and target from variable `{ev.target_var}`.")

        # --- Iteration Logic (Cross-validation or Runs) ---
        if ev.cross_validation:
            cv_type = ev.cross_validation['type']
            cv_n = ev.cross_validation['n']
            prompt_lines.append(f"\n# Perform training using {cv_n} {cv_type} {'splits' if cv_type == 'runs' else 'folds'}.")
            prompt_lines.append(f"# Inside each {'run' if cv_type == 'runs' else 'fold'}:")
            loop_indent = "  "
        else:
            prompt_lines.append("\n# Perform a single train-test split (e.g., 70/30 split, stratified if classification).")
            loop_indent = "" # No loop needed

        # --- Feature Selection (inside loop if applicable) ---
        if ev.feature_selection:
            fs_method = ev.feature_selection['method']
            k_values = ev.feature_selection['k_values']
            k_prompt = f"value {k_values}" if isinstance(k_values, int) else f"values {k_values}"
            prompt_lines.append(f"{loop_indent}- Perform feature selection using method '{fs_method}' with k {k_prompt}.")
            # Handle iterating through k_values if it's a list
            if isinstance(k_values, list):
                 prompt_lines.append(f"{loop_indent}- Iterate through each specified k value for feature selection.")
                 k_loop_indent = loop_indent + "  "
            else:
                 k_loop_indent = loop_indent # No extra loop for k
            prompt_lines.append(f"{k_loop_indent}- Select features from the training set of the current {'run' if cv_type == 'runs' else 'fold'}.")
            prompt_lines.append(f"{k_loop_indent}- Save the list of selected features (for this run/fold and k value) to a file using the prefix '{ev.save_path_prefix}_features_k{{k}}_run{{run_or_fold_index}}.txt'.")
            prompt_lines.append(f"{k_loop_indent}- Transform both training and test/validation sets using the selected features.")
            feature_set_var = "`X_train_fs`, `X_test_fs`" # Variables holding feature-selected data
        else:
            k_loop_indent = loop_indent
            feature_set_var = "`X_train`, `X_test`" # Original split data

        # --- Model Training (inside loops) ---
        prompt_lines.append(f"{k_loop_indent}- Train the following models on the feature set {feature_set_var}:")
        model_loop_indent = k_loop_indent + "  "
        for model_spec in ev.models:
            model_name = model_spec['name']
            params = model_spec.get('params', {})
            params_str = f" with parameters {params}" if params else ""
            prompt_lines.append(f"{model_loop_indent}- {model_name}{params_str}.")
            prompt_lines.append(f"{model_loop_indent}- Save the trained model (for this run/fold, k value, and model) to a file using the prefix '{ev.save_path_prefix}_model_{model_name}_k{{k}}_run{{run_or_fold_index}}.pkl'.")
            # --- Evaluation within loop ---
            prompt_lines.append(f"{model_loop_indent}- Evaluate the model on the corresponding test/validation set (`X_test_fs` or `X_test`).")
            prompt_lines.append(f"{model_loop_indent}- Calculate metrics: Accuracy, Precision, Recall, F1, AUC (if applicable).")
            prompt_lines.append(f"{model_loop_indent}- Store these metrics along with run/fold index, k value (if applicable), and model name.")
            if ev.output_roc_data_var:
                 prompt_lines.append(f"{model_loop_indent}- Calculate and store ROC curve data (FPR, TPR, AUC) for potential later plotting.")


        # --- Aggregation (outside loop) ---
        prompt_lines.append(f"\n# After all runs/folds are complete:")
        prompt_lines.append(f"- Aggregate the collected metrics across all runs/folds for each model and feature size (k).")
        prompt_lines.append(f"- Calculate mean and standard deviation for each metric (e.g., mean AUC, std AUC).")
        prompt_lines.append(f"- Store the aggregated metrics in a pandas DataFrame named `{ev.output_metrics_var}`.")
        prompt_lines.append(f"- Save this DataFrame to a CSV file: '{ev.save_path_prefix}_metrics_aggregated.csv'.")
        if ev.output_roc_data_var:
            prompt_lines.append(f"- Store the collected ROC data (from all runs) in a structure (e.g., dictionary) named `{ev.output_roc_data_var}`.")
            prompt_lines.append(f"- Save this ROC data structure to a pickle file: '{ev.save_path_prefix}_roc_data.pkl'.")
        prompt_lines.append(f"- Print the head of the aggregated metrics DataFrame `{ev.output_metrics_var}`.")

        if ev.additional_instructions:
            prompt_lines.append(f"\n# Additional User Instructions:\n{ev.additional_instructions}")

        prompt = "\n".join(prompt_lines)
        await self._execute_code_prompt(ctx, prompt, "Train Models")
        await ctx.set('last_metrics_var', ev.output_metrics_var)
        if ev.output_roc_data_var: await ctx.set('last_roc_data_var', ev.output_roc_data_var)

        print("Training complete.")
        print("What would you like to do next?")
        user_command = self._get_input("> ", required=True)
        return ML_Copilot_Command(user_command=user_command)

    @step
    async def evaluate(self, ctx: Context, ev: EvaluateEvent) -> ML_Copilot_Command:
        """Evaluates models based on the EvaluateEvent."""
        prompt_lines = ["Please write and execute Python code to evaluate model performance:"]

        prompt_lines.append(f"# Models/Results Source: {ev.models_source} (Interpret this as a variable name holding results, a file path pattern, or list of paths).")

        prompt_lines.append(f"\n# Evaluate on the following datasets:")
        for ds in ev.datasets:
             name = ds['name']
             features_var = ds['features_var']
             labels_var = ds.get('labels_var') # Optional for unsupervised or if labels are part of features_var
             survival_var = ds.get('survival_var') # Optional
             prompt_lines.append(f"- Dataset '{name}': Features=`{features_var}`" + (f", Labels=`{labels_var}`" if labels_var else "") + (f", Survival=`{survival_var}`" if survival_var else ""))

        prompt_lines.append(f"\n# For each model configuration found in the source:")
        prompt_lines.append(f"- Load the model if necessary.")
        prompt_lines.append(f"- For each specified validation dataset:")
        prompt_lines.append(f"  - Prepare the dataset features (handle common features if model was trained on different set).")
        prompt_lines.append(f"  - Make predictions.")
        if ev.save_path_prefix:
             prompt_lines.append(f"  - Save predictions to a file: '{ev.save_path_prefix}_predictions_{{model_config}}_{{dataset_name}}.csv'.")
        prompt_lines.append(f"  - Calculate the requested metrics: {', '.join(ev.metrics)}.")
        if 'LogRankPValue' in ev.metrics:
             if ev.time_col and ev.event_col:
                  prompt_lines.append(f"  - For LogRankPValue: Use survival data from the corresponding survival variable, time column '{ev.time_col}', event column '{ev.event_col}', grouping by predicted labels.")
             else:
                  prompt_lines.append(f"  - Warning: LogRankPValue requested but time/event columns not specified in event.")

        prompt_lines.append(f"\n# Aggregate the evaluation results across all models and datasets.")
        prompt_lines.append(f"- Create a summary pandas DataFrame named `{ev.output_summary_var}` containing model identifiers, dataset names, and calculated metrics.")
        if ev.save_path_prefix:
            prompt_lines.append(f"- Save this summary DataFrame to '{ev.save_path_prefix}_evaluation_summary.csv'.")
        prompt_lines.append(f"- Print the head of the summary DataFrame `{ev.output_summary_var}`.")

        if ev.additional_instructions:
            prompt_lines.append(f"\n# Additional User Instructions:\n{ev.additional_instructions}")

        prompt = "\n".join(prompt_lines)
        await self._execute_code_prompt(ctx, prompt, "Evaluate Models")
        await ctx.set('last_evaluation_summary_var', ev.output_summary_var)

        print("Evaluation complete.")
        print("What would you like to do next?")
        user_command = self._get_input("> ", required=True)
        return ML_Copilot_Command(user_command=user_command)

    @step
    async def select_model(self, ctx: Context, ev: SelectModelEvent) -> ML_Copilot_Command:
        """Selects the best model based on the SelectModelEvent."""
        prompt_lines = [f"Please write and execute Python code to select the best model based on criteria:"]
        prompt_lines.append(f"# Use the evaluation summary table stored in variable `{ev.summary_table_var}`.")

        prompt_lines.append(f"\n# Apply the following selection strategies sequentially until a model is found:")
        for i, strategy in enumerate(ev.criteria):
            name = strategy.get('name', f'Strategy_{i+1}')
            conditions = strategy.get('conditions', [])
            optimize = strategy.get('optimize', 'no_optimization') # e.g., 'maximize AUC', 'minimize p_value'
            prompt_lines.append(f"- Strategy '{name}':")
            if conditions:
                prompt_lines.append(f"  - Filter the summary table where: {' AND '.join(conditions)}.")
            prompt_lines.append(f"  - From the filtered results, select the best model by optimizing: {optimize}.")
            prompt_lines.append(f"  - If a model is found using this strategy, stop and proceed.")

        prompt_lines.append(f"\n# Store the configuration details of the selected best model (e.g., model name, feature size, run index, metrics) in a dictionary variable named `{ev.output_best_config_var}`.")
        # Include info needed to find the model file
        prompt_lines.append(f"- Ensure the dictionary includes enough information to reconstruct the model file path using the base path '{ev.models_base_path}'.")
        prompt_lines.append(f"- Print the selected best model configuration stored in `{ev.output_best_config_var}`.")

        if ev.additional_instructions:
            prompt_lines.append(f"\n# Additional User Instructions:\n{ev.additional_instructions}")

        prompt = "\n".join(prompt_lines)
        await self._execute_code_prompt(ctx, prompt, "Select Best Model")
        await ctx.set('last_best_model_config_var', ev.output_best_config_var)

        print("Model selection complete.")
        print("What would you like to do next?")
        user_command = self._get_input("> ", required=True)
        return ML_Copilot_Command(user_command=user_command)

    @step
    async def analyze_survival(self, ctx: Context, ev: AnalyzeSurvivalEvent) -> ML_Copilot_Command:
        """Performs survival analysis based on the AnalyzeSurvivalEvent."""
        prompt_lines = [f"Please write and execute Python code for survival analysis:"]
        prompt_lines.append(f"# Use the DataFrame stored in variable `{ev.input_df_var}`.")
        prompt_lines.append(f"# Time column: '{ev.time_col}', Event column: '{ev.event_col}', Group column: '{ev.group_col}'.")

        prompt_lines.append(f"\n# Perform Kaplan-Meier analysis:")
        prompt_lines.append(f"- Generate Kaplan-Meier curves for each group defined by the '{ev.group_col}' column.")
        prompt_lines.append(f"- Create a plot titled '{ev.plot_title}'.")
        prompt_lines.append(f"- Add appropriate labels and legend.")

        compare_str = json.dumps(ev.compare_groups) if isinstance(ev.compare_groups, list) else ev.compare_groups
        prompt_lines.append(f"- Perform log-rank tests to compare survival distributions between groups: {compare_str}.")
        prompt_lines.append(f"- Annotate the plot with the relevant log-rank p-value(s).")
        prompt_lines.append(f"- Save the plot to '{ev.plot_save_path}'.")

        if ev.results_save_path:
            prompt_lines.append(f"- Save the detailed log-rank test results (test statistic, p-value for each comparison) to '{ev.results_save_path}'.")

        if ev.additional_instructions:
            prompt_lines.append(f"\n# Additional User Instructions:\n{ev.additional_instructions}")

        prompt = "\n".join(prompt_lines)
        await self._execute_code_prompt(ctx, prompt, "Analyze Survival")

        print("Survival analysis complete.")
        print("What would you like to do next?")
        user_command = self._get_input("> ", required=True)
        return ML_Copilot_Command(user_command=user_command)


    @step
    async def plot(self, ctx: Context, ev: PlotEvent) -> ML_Copilot_Command:
        """Generates plots based on the PlotEvent."""
        prompt_lines = [f"Please write and execute Python code to generate a '{ev.plot_type}' plot:"]

        prompt_lines.append(f"\n# Required input data:")
        for key, var_name in ev.input_data.items():
            prompt_lines.append(f"- Use data for '{key}' from variable or path: `{var_name}`.")

        prompt_lines.append(f"\n# Apply the following plot parameters:")
        for key, value in ev.parameters.items():
            # Safely represent value in prompt
            value_repr = json.dumps(value) if not isinstance(value, (str, int, float, bool)) else repr(value)
            prompt_lines.append(f"- Set parameter '{key}' to {value_repr}.")

        # Add specific guidance based on plot type if needed
        if ev.plot_type == "ROC Curve":
             prompt_lines.append("- Ensure the plot includes mean ROC curves with standard deviation bands if multiple runs data is provided.")
             prompt_lines.append("- Include a diagonal chance line.")
        elif ev.plot_type == "AUC Boxplot":
             prompt_lines.append("- Create boxplots showing AUC distribution across runs, grouped by model and feature size.")
        # Add more plot-specific instructions here...

        prompt_lines.append(f"\n# Save the final plot to '{ev.save_path}'.")

        if ev.additional_instructions:
            prompt_lines.append(f"\n# Additional User Instructions:\n{ev.additional_instructions}")

        prompt = "\n".join(prompt_lines)
        await self._execute_code_prompt(ctx, prompt, f"Plotting ({ev.plot_type})")

        print(f"Plotting complete. Plot saved to {ev.save_path}")
        print("What would you like to do next?")
        user_command = self._get_input("> ", required=True)
        return ML_Copilot_Command(user_command=user_command)

    @step
    async def list_files(self, ctx: Context, ev: ListFilesEvent) -> ML_Copilot_Command:
        """Lists files in the current directory."""
        print("\n--- Files in Current Directory ---")
        try:
            files = os.listdir('.')
            for f in files:
                print(f"- {f}")
        except Exception as e:
            print(f"Error listing files: {e}")
        print("--------------------------------\n")
        print("What would you like to do next?")
        user_command = self._get_input("> ", required=True)
        return ML_Copilot_Command(user_command=user_command)

    @step
    async def custom_task(self, ctx: Context, ev: CustomTaskEvent) -> ML_Copilot_Command:
        """Executes a custom task based on the CustomTaskEvent."""
        prompt_lines = ["Please write and execute Python code for the following custom task:"]
        prompt_lines.append(f"\n# Task Description:\n{ev.instruction}")

        if ev.input_vars:
            prompt_lines.append("\n# Use the following input variables:")
            for desc, var_name in ev.input_vars.items():
                prompt_lines.append(f"- `{var_name}` (Represents: {desc})")

        if ev.output_description:
            prompt_lines.append(f"\n# Expected Output:\n{ev.output_description}")
        else:
            prompt_lines.append("\n# Perform the task and print any relevant results or confirmation.")

        prompt = "\n".join(prompt_lines)
        await self._execute_code_prompt(ctx, prompt, "Custom Task")

        print("Custom task execution attempted.")
        print("What would you like to do next?")
        user_command = self._get_input("> ", required=True)
        return ML_Copilot_Command(user_command=user_command)

    @step
    async def stop_workflow(self, ctx: Context, ev: StopWorkflowEvent) -> None:
        """Handles the stop event."""
        print(f"\n--- Workflow Terminated ---")
        print(ev.result)
        print("---------------------------\n")
        # No return value needed for the final step

# ===========================================================
# 3. Workflow Execution
# ===========================================================
async def main():
    # Configure Settings.llm here if not done globally
    # from llama_index.llms.openai import OpenAI
    # Settings.llm = OpenAI(model="gpt-4o") # Or your preferred model

    if not Settings.llm:
        print("ERROR: LLM is not configured in llama_index.core.Settings.")
        print("Please set Settings.llm before running the workflow.")
        print("Example: from llama_index.llms.openai import OpenAI; from llama_index.core import Settings; Settings.llm = OpenAI(model='gpt-4o')")
        return

    workflow = MLWorkflow(timeout=1200, verbose=True) # Increased timeout for complex tasks
    # You can add event handlers here if needed
    # workflow.on(StartEvent)(lambda ev: print("Workflow started"))
    # workflow.on(StopWorkflowEvent)(lambda ev: print(f"Workflow stopped with result: {ev.result}"))
    await workflow.run()

if __name__ == "__main__":
    # IMPORTANT: Make sure your OpenAI API key is set as an environment variable
    # export OPENAI_API_KEY='your-api-key'
    # Also ensure necessary libraries (pandas, scikit-learn, lifelines, etc.) are installed
    # pip install llama-index llama-index-agent-openai llama-index-tools-code-interpreter pandas scikit-learn matplotlib lifelines
    asyncio.run(main())