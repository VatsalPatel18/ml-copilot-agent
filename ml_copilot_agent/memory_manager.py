# ml_copilot_agent/memory_manager.py

import os
import json
import logging
from typing import Optional

from llama_index.core.workflow import Context, JsonSerializer, JsonPickleSerializer
from llama_index.core.base.llms.base import BaseLLM
from llama_index.core.base.embeddings.base import BaseEmbedding

from .config import STATE_FILENAME

logger = logging.getLogger(__name__)

class MemoryManager:
    """Handles loading and saving of the workflow context (state/memory)."""

    def __init__(self, project_path: str):
        """
        Initializes the MemoryManager for a specific project.

        Args:
            project_path: The root directory of the current project.
        """
        self.state_file_path = os.path.join(project_path, STATE_FILENAME)
        # Use JsonPickleSerializer if complex objects might be stored in context,
        # otherwise JsonSerializer is safer and more portable.
        self.serializer = JsonSerializer()
        # self.serializer = JsonPickleSerializer()
        logger.info(f"Memory manager initialized for project path: {project_path}")
        logger.info(f"State file path: {self.state_file_path}")

    def load_context(self, llm: Optional[BaseLLM] = None, embed_model: Optional[BaseEmbedding] = None) -> Context:
        """
        Loads the workflow context from the state file, or creates a new one.

        Args:
            llm: The initialized LLM (might be needed if context depends on it).
            embed_model: The initialized Embedding model (might be needed).

        Returns:
            The loaded or newly created Context object.
        """
        if os.path.exists(self.state_file_path):
            try:
                with open(self.state_file_path, 'r') as f:
                    ctx_dict = json.load(f)
                # We need a workflow instance to restore context, but the workflow
                # itself needs the context. This creates a circular dependency.
                # Workaround: Create a dummy context first, load state into it.
                # The workflow will later use this pre-loaded context.
                # Alternatively, the workflow could load the context itself after initialization.
                # Let's assume the workflow handles Context restoration internally or we pass the dict.
                # For simplicity here, we'll just return a basic Context if loading fails.
                # A better approach might be needed depending on AgentWorkflow's Context requirements.

                # Placeholder: AgentWorkflow might need specific setup for context restoration.
                # This part might need adjustment based on how AgentWorkflow handles loading.
                # For now, we just log and return a new context if loading fails.
                # A more robust way would be to pass ctx_dict to the workflow constructor.
                logger.info(f"Loaded context dictionary from {self.state_file_path}")
                # We can't fully restore without the workflow instance here.
                # Let's return a new Context and let the workflow populate it from the dict if needed.
                # Or, store essential state parts directly.
                # Let's store the 'state' dictionary part if it exists.
                initial_state_data = ctx_dict.get("state", {})
                ctx = Context()
                ctx.set("state", initial_state_data) # Pre-populate state
                logger.info("Created new context and populated 'state' from loaded file.")
                return ctx

            except (json.JSONDecodeError, OSError, KeyError) as e:
                logger.error(f"Error loading or parsing state file {self.state_file_path}: {e}. Creating new context.")
                # Fall through to create a new context
            except Exception as e:
                 logger.exception(f"Unexpected error loading state file {self.state_file_path}: {e}. Creating new context.")
                 # Fall through
        else:
            logger.info("State file not found. Creating new context.")

        # Create a new context if loading failed or file doesn't exist
        ctx = Context()
        # Initialize the 'state' dictionary within the context
        ctx.set("state", {
            "project_name": os.path.basename(os.path.dirname(self.state_file_path)), # Store project name
            "llm_info": llm.metadata.model_name if llm else "Unknown", # Store LLM info
             # Add other initial state variables as needed
            "raw_data_path": None,
            "preprocessed_data_path": None,
            "target_column": None,
            "feature_columns": None,
            "sample_id_column": None, # Important for saving predictions
            "trained_models": {}, # Dict to store info about trained models {model_id: info}
            "evaluation_results": {}, # Dict to store eval results {model_id: results}
            "selected_features": {}, # Dict to store selected features {model_id: features}
            "plots": [], # List of generated plot file paths
            "log_rag_index_exists": False,
            "learn_rag_index_exists": False,
            "learn_rag_files": [],
        })
        logger.info("Initialized new workflow context with default state.")
        return ctx

    def save_context(self, context: Context):
        """
        Saves the workflow context to the state file.

        Args:
            context: The Context object to save.
        """
        try:
            # AgentWorkflow context might not have a direct to_dict method like the base Workflow context.
            # We need to extract the relevant state manually or use the intended mechanism.
            # Let's assume the core state is stored under the key "state".
            state_data = context.get("state", default={})

            # Use a temporary dictionary if direct serialization isn't available/safe
            ctx_dict_to_save = {"state": state_data} # Save only the state part for now

            # Alternative: If context has to_dict() and works with AgentWorkflow
            # ctx_dict_to_save = context.to_dict(serializer=self.serializer)

            os.makedirs(os.path.dirname(self.state_file_path), exist_ok=True)
            with open(self.state_file_path, 'w') as f:
                json.dump(ctx_dict_to_save, f, indent=4) # Use standard json dump for the state dict
            logger.info(f"Successfully saved context state to {self.state_file_path}")
        except Exception as e:
            logger.exception(f"Error saving context state to {self.state_file_path}: {e}")
            print(f"Warning: Could not save project state - {e}")
