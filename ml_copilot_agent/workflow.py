# ml_copilot_agent/workflow.py

import os
import logging
import traceback
import datetime # Added import
from typing import List, Dict, Any, Optional

# LlamaIndex imports
from llama_index.core.agent.workflow import AgentWorkflow, FunctionAgent
# Removed unused event imports like AgentInput, AgentAction, etc. as we rely on the agent's response
from llama_index.core.workflow import Context # Keep Context for state management
from llama_index.core.base.llms.base import BaseLLM
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.tools import BaseTool, FunctionTool
# Removed base Workflow, StartEvent, StopEvent, step, Event as AgentWorkflow handles the flow

# Project components
from .config import (DEFAULT_PREPROCESS_SAVE_DIR, DEFAULT_MODEL_SAVE_DIR,
                     DEFAULT_RESULTS_SAVE_DIR, DEFAULT_PLOT_SAVE_DIR,
                     DEFAULT_PREPROCESS_FILENAME, DEFAULT_MODEL_FILENAME,
                     DEFAULT_EVALUATION_FILENAME)
from .log_manager import LogManager
from .memory_manager import MemoryManager
from .rag_manager import RAGManager
from .tools import (get_code_interpreter_tool, get_list_files_tool,
                    get_log_query_tool, get_learn_query_tool, get_learn_setup_tool)

logger = logging.getLogger(__name__)

# --- Custom Events Removed ---
# In this AgentWorkflow setup with a single agent, the flow is driven by the
# agent's interpretation of the user message and state, rather than explicit
# event transitions between defined @steps. Custom events are not needed here.


# --- Main Agent Workflow ---
class MLCopilotWorkflow(AgentWorkflow):
    """
    The main AgentWorkflow orchestrating the ML Copilot tasks.
    Uses a single, versatile agent that interprets user requests, gathers info,
    generates code, and uses tools.
    """

    def __init__(
        self,
        llm: BaseLLM,
        embed_model: BaseEmbedding,
        log_manager: LogManager,
        memory_manager: MemoryManager,
        project_path: str,
        initial_state: Optional[Dict[str, Any]] = None,
        verbose: bool = False,
        **kwargs,
    ):
        """
        Initializes the ML Copilot Agent Workflow.

        Args:
            llm: The language model instance.
            embed_model: The embedding model instance.
            log_manager: Instance for logging.
            memory_manager: Instance for managing state persistence.
            project_path: Path to the current project.
            initial_state: Initial state dictionary loaded from memory.
            verbose: Whether to enable verbose logging/output.
            **kwargs: Additional arguments for AgentWorkflow.
        """
        self.log_manager = log_manager
        self.memory_manager = memory_manager
        self.project_path = project_path
        # Ensure RAGManager gets all required args
        self.rag_manager = RAGManager(project_path, embed_model, llm, log_manager)

        # --- Define Tools ---
        self.log_manager.log("INFO", "Initializing tools...")
        code_tool = get_code_interpreter_tool(log_manager)
        list_files_tool = get_list_files_tool(project_path, log_manager)
        log_query_tool = get_log_query_tool(log_manager, self.rag_manager)
        learn_query_tool = get_learn_query_tool(log_manager, self.rag_manager)
        learn_setup_tool = get_learn_setup_tool(log_manager, self.rag_manager)

        tools: List[BaseTool] = [
            code_tool,
            list_files_tool,
            log_query_tool,
            learn_query_tool,
            learn_setup_tool,
            # Add more tools as needed
        ]
        tool_names = [t.metadata.name for t in tools]
        self.log_manager.log("INFO", f"Tools initialized: {tool_names}")

        # --- Define the Agent ---
        # Using a single FunctionAgent. The system prompt is CRITICAL for guiding its behavior.
        system_prompt = f"""
You are ML Copilot Agent, an AI assistant for machine learning tasks within project: '{os.path.basename(project_path)}'.
Project Path on Server: {self.project_path} (Use this for context, but generate code that saves outputs to relative subdirs like 'data/', 'models/', 'results/', 'plots/').

**Your Capabilities & Tools:**
* **Code Execution (`{code_tool.metadata.name}`):** Execute Python code for data loading, preprocessing, training, evaluation, plotting. MUST generate complete, runnable scripts including imports (pandas, sklearn, etc.) and saving results/models/plots to project subdirectories.
* **List Files (`{list_files_tool.metadata.name}`):** List contents of project subdirectories (data, models, results, plots, rag_data, or '.' for root).
* **Query Logs (`{log_query_tool.metadata.name}`):** Answer questions about past actions and results using project logs.
* **Setup Learning (`{learn_setup_tool.metadata.name}`):** Load and index user-provided documents (PDFs, text files) for learning. Requires a list of file paths.
* **Query Learned Docs (`{learn_query_tool.metadata.name}`):** Answer questions based on documents loaded via Setup Learning.

**Core Workflow:**
1.  **Understand Goal:** Identify the user's primary task (preprocess, train, plot, list files, custom code, check logs, learn new docs, query learned docs).
2.  **Gather Info (Iteratively):** If needed, ask specific, sequential questions to get required details. Examples:
    * *Preprocess:* Ask for dataset path, target column name, desired save path (use default '{DEFAULT_PREPROCESS_SAVE_DIR}/{DEFAULT_PREPROCESS_FILENAME}' if none), specific instructions (scaling method, missing value strategy).
    * *Train:* Ask for preprocessed data path (use state if available), task type (classification/regression), classification details (binary/multi-class/multi-label), model type(s) (e.g., 'LogisticRegression', 'RandomForest', list like ['SVM', 'KNN']), model save path (use default '{DEFAULT_MODEL_SAVE_DIR}/{DEFAULT_MODEL_FILENAME}'), evaluation requirements (e.g., cross-validation folds, specific metrics beyond defaults), feature selection needs. **Evaluation is part of training.**
    * *Plot:* Ask what to plot (e.g., 'evaluation results', 'data features'), data source path (use state if relevant, e.g., preprocessed data or evaluation results file), specific plot types (e.g., 'correlation matrix', 'ROC curve', 'feature distribution'), desired save directory (use default '{DEFAULT_RESULTS_SAVE_DIR}/{DEFAULT_PLOT_SAVE_DIR}').
    * *Custom Code:* Ask for the specific Python code or detailed instructions.
    * *Check Logs:* Ask for the specific question about the logs.
    * *Learn New Docs:* Ask for the list of file paths to index OR the question to ask about already indexed docs.
3.  **Select Tool:** Choose the most appropriate tool based on the goal.
4.  **Prepare Tool Input:**
    * *Code Interpreter:* Generate the complete Python script. Use information from the current state (like `preprocessed_data_path` for training). Ensure code saves outputs correctly to relative paths (e.g., `os.path.join('{DEFAULT_MODEL_SAVE_DIR}', 'my_model.pkl')`). Include error handling (try-except) in the generated code where appropriate.
    * *Other Tools:* Format arguments as required by the tool description (e.g., `sub_directory='data'` for list_files, `query='show errors from last run'` for query_logs).
5.  **Execute Tool:** Call the selected tool.
6.  **Process Result & Update State:** Analyze the tool output. Summarize the outcome for the user. *Crucially, recognize when actions create or modify key files (like preprocessed data or models) and implicitly update your understanding of the project state for future steps.* (e.g., If preprocessing saves to 'data/preprocessed.csv', remember this path).
7.  **Respond & Prompt:** Inform the user of the result (e.g., "Preprocessing complete, saved to data/preprocessed.csv"). If the task is done, ask what they want to do next.

**State Management:**
* You have access to the conversation history and the current state implicitly.
* Key state information you need to track includes: `raw_data_path`, `preprocessed_data_path`, `target_column`, `feature_columns`, `sample_id_column`, details of `trained_models` (path, type, features used), `evaluation_results` paths, generated `plots` paths, `log_rag_index_exists`, `learn_rag_index_exists`, `learn_rag_files`.
* Use this state information when generating code for subsequent steps (e.g., use `preprocessed_data_path` when training).

**Current Project State (Summary):**
Raw Data: {initial_state.get('raw_data_path', 'Not set')}
Preprocessed Data: {initial_state.get('preprocessed_data_path', 'Not set')}
Target Column: {initial_state.get('target_column', 'Not set')}
Models Trained: {list(initial_state.get('trained_models', {}).keys())}
Log RAG Ready: {initial_state.get('log_rag_index_exists', False)}
Learn RAG Ready: {initial_state.get('learn_rag_index_exists', False)} (Files: {len(initial_state.get('learn_rag_files', []))})
"""

        ml_agent = FunctionAgent(
            name="MLCopilotAgent",
            description="Handles machine learning tasks, file operations, logging, and RAG within a project.",
            system_prompt=system_prompt,
            llm=llm,
            tools=tools,
            verbose=verbose,
            # No handoff needed for single agent workflow
        )

        # Initialize the AgentWorkflow
        super().__init__(
            agents=[ml_agent], # Single agent for now
            root_agent=ml_agent.name,
            initial_state=initial_state if initial_state else {}, # Pass the loaded state
            llm=llm, # Pass LLM for potential internal use by AgentWorkflow
            verbose=verbose,
            timeout=kwargs.get('timeout', 600),
            **kwargs,
        )
        self.log_manager.log("INFO", "ML Copilot AgentWorkflow initialized.")

    # Override run to potentially add pre/post processing or custom event handling if needed
    async def run(self, *args, ctx: Optional[Context] = None, **kwargs) -> Any:
        """
        Runs the agent workflow for one turn.

        Args:
            *args: Positional arguments, typically the user message.
            ctx: The workflow context containing state. If None, it will be loaded.
            **kwargs: Keyword arguments for the workflow run.

        Returns:
            The final response object from the agent for this turn.
        """
        user_msg = kwargs.get('user_msg', args[0] if args else None)
        turn_start_time = datetime.datetime.now(datetime.timezone.utc)

        if user_msg:
             self.log_manager.log("INFO", "Received user input", data={"message": user_msg})

        if ctx is None:
             # This shouldn't happen if managed by __main__, but as a fallback:
             logger.warning("Context not provided to workflow run, attempting to load.")
             # Pass LLM/embed_model if they are stored attributes or accessible
             ctx = self.memory_manager.load_context(llm=self.llm, embed_model=self.embed_model) # Ensure self.llm/self.embed_model exist

        # Update context with current timestamp or other pre-run info if needed
        current_state = await ctx.get("state", default={}) # Use await for async context access
        current_state["last_interaction_time"] = turn_start_time.isoformat()
        # Add current state summary to prompt dynamically if possible/needed
        # (This might require modifying how the agent's chat history/prompt is built)

        await ctx.set("state", current_state) # Use await

        try:
            # The core execution is handled by the parent AgentWorkflow.run
            result = await super().run(*args, ctx=ctx, **kwargs) # Pass context

            # Log the final output of the turn
            # result is likely an AgentResponse object
            final_response_text = ""
            if hasattr(result, 'response') and result.response:
                 final_response_text = str(result.response)
            elif isinstance(result, str): # Fallback if it's just a string
                 final_response_text = result
            else:
                 final_response_text = str(result) # Best guess

            self.log_manager.log("INFO", "Agent produced final response", data={"response_snippet": final_response_text[:500] + "..."})

            # --- State Update ---
            # AgentWorkflow *should* manage the state within the context. Tools
            # designed to accept `ctx` can also modify it using `await ctx.set(...)`.
            # Explicit updates here might be redundant or conflict if not careful.
            # We rely on the agent/tools updating the context passed to super().run().
            # We retrieve the final state primarily for logging/debugging here.
            final_state = await ctx.get("state", default={})
            self.log_manager.log("DEBUG", "State after agent run", data=final_state)

            # --- Persistence ---
            # Saving happens in __main__ after the run completes.

            return result # Return the agent's final response object

        except Exception as e:
            logger.exception("Error during AgentWorkflow run")
            self.log_manager.log("ERROR", f"AgentWorkflow run failed: {e}", data={"user_input": user_msg}, exc_info=True)
            # Save context even on error - saving happens in __main__ finally block
            # self.memory_manager.save_context(ctx) # Avoid saving here, let __main__ handle it
            # Return a user-friendly error message
            return f"An error occurred while processing your request: {e}. Please check the logs for details or try rephrasing your request."
