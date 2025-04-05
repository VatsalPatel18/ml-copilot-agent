# ml_copilot_agent/__main__.py

import asyncio
import os
from dotenv import load_dotenv
import logging

from .project_manager import ProjectManager
from .llm_manager import LLMManager
from .workflow import MLCopilotWorkflow
from .memory_manager import MemoryManager
from .log_manager import LogManager, configure_logging
from .config import PROJECTS_DIR, LOG_LEVEL

# Configure logging at the entry point
configure_logging(level=LOG_LEVEL)
logger = logging.getLogger(__name__)

async def run_agent():
    """Main async function to run the ML Copilot Agent."""
    load_dotenv()
    logger.info("--- ML Copilot Agent Initializing ---")

    project_manager = ProjectManager(PROJECTS_DIR)
    llm_manager = LLMManager()

    # --- Project Selection/Creation ---
    project_name = project_manager.select_or_create_project()
    if not project_name:
        logger.error("No project selected or created. Exiting.")
        return
    logger.info(f"Working on project: {project_name}")

    project_path = project_manager.get_project_path(project_name)
    log_manager = LogManager(project_path) # Initialize logger for the specific project
    memory_manager = MemoryManager(project_path)

    # --- LLM Configuration ---
    llm, embed_model = await llm_manager.configure_llm()
    if not llm:
        logger.error("LLM configuration failed. Exiting.")
        log_manager.log("ERROR", "LLM configuration failed.")
        return
    logger.info(f"Using LLM: {llm.metadata.model_name}")
    log_manager.log("INFO", f"LLM configured: {llm.metadata.model_name}")

    # --- Load or Initialize Workflow Context ---
    logger.info("Loading/Initializing workflow context...")
    workflow_context = memory_manager.load_context(llm=llm, embed_model=embed_model) # Pass LLM/Embed model if needed by context/memory
    initial_state = workflow_context.get("state", default={}) # Get initial state if exists
    logger.debug(f"Initial state loaded: {initial_state}")

    # --- Initialize and Run Workflow ---
    logger.info("Initializing Agent Workflow...")
    # Pass necessary managers/configs to the workflow
    workflow = MLCopilotWorkflow(
        llm=llm,
        embed_model=embed_model,
        log_manager=log_manager,
        memory_manager=memory_manager,
        project_path=project_path,
        initial_state=initial_state,
        verbose=True # Or get from config
    )

    logger.info("--- Starting Agent Interaction ---")
    print("\nWelcome to the ML Copilot Agent!")
    print(f"Project: {project_name}")
    print("Type 'exit' or 'quit' to end the session.")

    try:
        # AgentWorkflow doesn't have a persistent loop like the old workflow.
        # We need to manage the interaction loop here.
        while True:
            user_input = input("\nWhat would you like to do next? \n> ")
            if user_input.lower() in ["exit", "quit"]:
                logger.info("User requested exit.")
                break

            # Run the workflow for one turn
            # The workflow itself should handle the state update via the context
            response = await workflow.run(user_msg=user_input, ctx=workflow_context)

            # Print the final response for this turn
            print(f"\nAssistant:\n{response}")

            # Save context after each turn
            logger.info("Saving workflow context...")
            memory_manager.save_context(workflow_context)
            log_manager.log("INFO", "Workflow context saved.")

    except KeyboardInterrupt:
        logger.warning("Keyboard interrupt received.")
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")
        log_manager.log("ERROR", f"Workflow crashed: {e}", exc_info=True)
    finally:
        # Ensure context is saved on exit
        logger.info("Final context save before exiting.")
        memory_manager.save_context(workflow_context)
        log_manager.log("INFO", "Final context saved on exit.")
        print("\nExiting ML Copilot Agent. Goodbye!")
        logger.info("--- ML Copilot Agent Session Ended ---")


def main():
    """Synchronous entry point."""
    # nest_asyncio allows running asyncio loops within existing ones (useful in some environments)
    # import nest_asyncio
    # nest_asyncio.apply()
    try:
        asyncio.run(run_agent())
    except Exception as e:
        logger.critical(f"Critical error during agent execution: {e}", exc_info=True)

if __name__ == "__main__":
    main()