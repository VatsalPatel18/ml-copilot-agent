# ml_copilot_agent/__main__.py

import asyncio
import sys
import os
import argparse
import traceback # Import traceback

# Core workflow components (adjust imports if needed)
try:
    from .workflow import MLWorkflow
    from . import initialize
    from llama_index.core.memory import ChatMemoryBuffer
    from llama_index.core.utils import get_tokenizer
except ImportError:
    print("Attempting imports assuming script execution context...")
    from workflow import MLWorkflow
    import initialize as init_module # Use alias to avoid name clash
    from llama_index.core.memory import ChatMemoryBuffer
    from llama_index.core.utils import get_tokenizer
    # Define initialize function if running as script and it's expected here
    if not hasattr(init_module, 'initialize'):
         print("Error: 'initialize' function not found.")
         sys.exit(1)
    initialize = init_module.initialize


def main():
    parser = argparse.ArgumentParser(description="ML Copilot Agent")
    parser.add_argument(
        "api_key",
        nargs='?', # Make API key optional if env var is set
        help="Your OpenAI API key (optional if OPENAI_API_KEY env var is set)"
    )
    # Removed checkpoint arguments

    args = parser.parse_args()

    # Determine API Key
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")

    if not api_key:
        print("Error: OpenAI API key not provided.")
        print("Please provide the key as a command-line argument or set the OPENAI_API_KEY environment variable.")
        sys.exit(1)

    # --- Initialization ---
    try:
        initialize(api_key)
        print("LLM Settings Initialized.")
    except Exception as e:
        print(f"Error during LLM initialization: {e}")
        sys.exit(1)

    # --- Memory Setup ---
    try:
        tokenizer = get_tokenizer()
        print("Tokenizer loaded for memory management.")
    except ImportError:
        print("Warning: Default tokenizer not found. Using basic length check for memory.")
        tokenizer = len # Fallback to simple length
    # Consider making token limit configurable via args if needed
    memory = ChatMemoryBuffer.from_defaults(token_limit=16000, tokenizer_fn=tokenizer)
    print("Chat Memory Buffer Initialized.")

    # --- Workflow Setup ---
    try:
        # Pass the initialized memory to the workflow
        workflow = MLWorkflow(memory=memory, timeout=1200, verbose=True)
        print("ML Workflow Initialized.")
    except Exception as e:
        print(f"Error initializing ML Workflow: {e}")
        traceback.print_exc()
        sys.exit(1)

    # --- Run Workflow ---
    try:
        print(f"\nStarting ML Copilot...")
        # Run the workflow directly
        asyncio.run(workflow.run())

    except KeyboardInterrupt:
        print("\nWorkflow interrupted by user. Exiting.")
    except Exception as e:
         print(f"\nAn unexpected error occurred during workflow execution: {e}")
         traceback.print_exc() # Print full traceback for debugging

if __name__ == "__main__":
    main()
