# ml_copilot_agent/tools.py

import os
import logging
import pandas as pd
import traceback
from typing import Dict, Any, Optional

# LlamaIndex imports
from llama_index.core.tools import FunctionTool
from llama_index.core.tools.tool_spec.base import BaseToolSpec
from llama_index.tools.code_interpreter import CodeInterpreterToolSpec # Replaces LocalCodeInterpreter

# Configuration and Managers
from .config import (DEFAULT_PREPROCESS_SAVE_DIR, DEFAULT_MODEL_SAVE_DIR,
                     DEFAULT_RESULTS_SAVE_DIR, DEFAULT_PLOT_SAVE_DIR)
from .log_manager import LogManager

logger = logging.getLogger(__name__)

# --- Code Execution Tool ---

# Option 1: Using LlamaIndex's built-in CodeInterpreterToolSpec (Recommended)
# This provides a sandboxed environment.
def get_code_interpreter_tool(log_manager: LogManager) -> FunctionTool:
    """
    Creates a FunctionTool wrapping the CodeInterpreterToolSpec.
    """
    try:
        code_spec = CodeInterpreterToolSpec()
        # The spec provides multiple tools, we might want to wrap the primary one
        # Or let the agent choose from the list provided by code_spec.to_tool_list()
        # Let's wrap the main 'code_interpreter' tool if available, otherwise raise error.

        # code_spec.to_tool_list() returns a list of FunctionTool objects
        tool_list = code_spec.to_tool_list()
        interpreter_tool = next((t for t in tool_list if t.metadata.name == 'code_interpreter'), None)

        if not interpreter_tool:
             raise ValueError("Could not find 'code_interpreter' tool in CodeInterpreterToolSpec")

        # We can potentially wrap the tool's function to add logging
        original_sync_fn = interpreter_tool.fn
        original_async_fn = interpreter_tool.async_fn

        async def logged_async_code_interpreter(*args, **kwargs):
            code_to_run = kwargs.get('code', args[0] if args else None) # Adjust based on actual signature
            log_manager.log("ACTION", "Executing code via interpreter", data={"code_snippet": code_to_run[:500] + "..." if code_to_run else "N/A"}) # Log snippet
            try:
                result = await original_async_fn(*args, **kwargs)
                # LlamaIndex code interpreter result is often an AgentChatResponse or similar object
                # We need to extract the relevant output (stdout, stderr, result data)
                output_str = str(result) # Basic string representation
                # TODO: Parse result object for structured output if possible
                log_manager.log("SUCCESS", "Code execution successful", data={"output_snippet": output_str[:500] + "..."})
                return result
            except Exception as e:
                tb_str = traceback.format_exc()
                log_manager.log("ERROR", f"Code execution failed: {e}", data={"traceback": tb_str})
                # Re-raise the exception so the agent knows it failed
                raise e

        # Create a new tool with the wrapped function
        # Note: Creating a new FunctionTool might lose some metadata from the original.
        # It might be better to modify the existing tool's function if possible,
        # but that's harder. Let's create a new one for simplicity.
        logged_tool = FunctionTool.from_defaults(
            fn=None, # Sync version not wrapped here for brevity, focus on async
            async_fn=logged_async_code_interpreter,
            name="logged_code_interpreter", # Give it a distinct name
            description="Executes Python code securely. Use this for data analysis, model training, plotting, etc. Provide complete, runnable code.",
            # Copy other relevant metadata if needed
        )
        logger.info("Code Interpreter tool created successfully.")
        return logged_tool

    except Exception as e:
        logger.exception(f"Failed to initialize CodeInterpreterToolSpec: {e}")
        print(f"Error: Failed to initialize Code Interpreter Tool - {e}")
        # Return a dummy tool or raise error
        return FunctionTool.from_defaults(fn=lambda *args, **kwargs: "Error: Code Interpreter Tool not available.", name="error_tool", description="Reports tool initialization error.")


# --- RAG Tools (Placeholders) ---

def get_log_query_tool(log_manager: LogManager, rag_manager) -> FunctionTool:
    """
    Creates a tool to query project logs using RAG. (Placeholder)
    """
    async def query_logs(query: str) -> str:
        log_manager.log("ACTION", "Querying project logs", data={"query": query})
        try:
            # 1. Ensure log index exists (call rag_manager.ensure_log_index())
            # 2. Perform query (call rag_manager.query_log_index(query))
            # result = await rag_manager.query_log_index(query) # Replace with actual call
            result = "Log querying not fully implemented yet." # Placeholder
            log_manager.log("SUCCESS", "Log query successful", data={"response_snippet": result[:200]})
            return result
        except Exception as e:
            tb_str = traceback.format_exc()
            log_manager.log("ERROR", f"Log query failed: {e}", data={"query": query, "traceback": tb_str})
            return f"Error querying logs: {e}"

    return FunctionTool.from_defaults(
        fn=None, # Async only for now
        async_fn=query_logs,
        name="query_project_logs",
        description="Queries the history of actions, errors, and results for the current project based on a natural language query."
    )

def get_learn_query_tool(log_manager: LogManager, rag_manager) -> FunctionTool:
    """
    Creates a tool to query documents loaded for learning using RAG. (Placeholder)
    """
    async def query_learned_docs(query: str) -> str:
        log_manager.log("ACTION", "Querying learned documents", data={"query": query})
        try:
            # 1. Check if learn index exists (use state from context via rag_manager or memory?)
            # 2. Perform query (call rag_manager.query_learn_index(query))
            # result = await rag_manager.query_learn_index(query) # Replace with actual call
            result = "Document querying (Learn New Things) not fully implemented yet." # Placeholder
            log_manager.log("SUCCESS", "Learned document query successful", data={"response_snippet": result[:200]})
            return result
        except Exception as e:
            tb_str = traceback.format_exc()
            log_manager.log("ERROR", f"Learned document query failed: {e}", data={"query": query, "traceback": tb_str})
            return f"Error querying learned documents: {e}"

    return FunctionTool.from_defaults(
        fn=None, # Async only for now
        async_fn=query_learned_docs,
        name="query_learned_documents",
        description="Answers questions based on the documents previously loaded using the 'Learn Something New' feature."
    )

def get_learn_setup_tool(log_manager: LogManager, rag_manager) -> FunctionTool:
     """
     Creates a tool to setup RAG for learning new documents. (Placeholder)
     """
     async def setup_learning_rag(file_paths: list[str]) -> str:
         log_manager.log("ACTION", "Setting up RAG for learning", data={"files": file_paths})
         try:
            # 1. Validate file paths
            # 2. Load documents (rag_manager.load_documents(file_paths))
            # 3. Build/update index (rag_manager.build_learn_index(documents))
            # 4. Update project state (learn_rag_index_exists=True, learn_rag_files=...)
            # result = await rag_manager.setup_learning_rag(file_paths) # Replace with actual call
            result = f"Setup for learning RAG initiated for {len(file_paths)} files. (Implementation Pending)" # Placeholder
            log_manager.log("SUCCESS", "Learning RAG setup successful", data={"files": file_paths})
            return result
         except Exception as e:
            tb_str = traceback.format_exc()
            log_manager.log("ERROR", f"Learning RAG setup failed: {e}", data={"files": file_paths, "traceback": tb_str})
            return f"Error setting up learning RAG: {e}"

     return FunctionTool.from_defaults(
        fn=None, # Async only
        async_fn=setup_learning_rag,
        name="setup_learning_rag",
        description="Loads and indexes the specified PDF or text files to enable querying them via 'query_learned_documents'."
     )


# --- Other Potential Tools ---
# Example: A tool to explicitly list files in a project directory

def get_list_files_tool(project_path: str, log_manager: LogManager) -> FunctionTool:
    """Creates a tool to list files in project subdirectories."""
    def list_files(sub_directory: str = ".") -> str:
        """
        Lists files and directories within a specified project subdirectory (e.g., 'data', 'models', 'results', 'plots').
        Defaults to the main project directory if no subdirectory is specified.
        """
        log_manager.log("ACTION", f"Listing files in subdirectory: '{sub_directory}'")
        allowed_dirs = ["data", "models", "results", "plots", "rag_data", "."] # Allow listing in base dir too
        target_subdir = sub_directory.strip().strip('/')

        if target_subdir not in allowed_dirs and target_subdir != "":
             log_manager.log("WARN", f"Attempted to list forbidden directory: {target_subdir}")
             return f"Error: Can only list files in allowed subdirectories: {allowed_dirs}"

        if target_subdir == ".":
            target_path = project_path
        else:
            target_path = os.path.join(project_path, target_subdir)


        if not os.path.isdir(target_path):
            log_manager.log("WARN", f"Subdirectory not found: {target_path}")
            return f"Error: Subdirectory '{target_subdir}' not found in project."

        try:
            items = os.listdir(target_path)
            if not items:
                result = f"Directory '{target_subdir}' is empty."
            else:
                result = f"Contents of '{target_subdir}':\n- " + "\n- ".join(items)
            log_manager.log("SUCCESS", f"Listed files in '{target_subdir}'", data={"item_count": len(items)})
            return result
        except OSError as e:
            log_manager.log("ERROR", f"Error listing files in {target_path}: {e}")
            return f"Error listing files in '{target_subdir}': {e}"

    return FunctionTool.from_defaults(
        fn=list_files,
        name="list_project_files",
        description="Lists files and directories within a specified project subdirectory (data, models, results, plots, rag_data, or '.' for project root)."
    )
