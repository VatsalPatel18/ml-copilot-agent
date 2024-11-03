# ml_copilot_agent/__main__.py

import asyncio
import sys
import os

from .workflow import MLWorkflow
from . import initialize

def main():
    # Check for the API key in command-line arguments or environment variable
    if len(sys.argv) > 1:
        api_key = sys.argv[1]
    else:
        api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("Usage: python -m ml_copilot_agent <OPENAI_API_KEY> or set the OPENAI_API_KEY environment variable.")
        sys.exit(1)
    
    initialize(api_key)

    async def run_workflow():
        workflow = MLWorkflow(timeout=600, verbose=True)
        await workflow.run()
    
    asyncio.run(run_workflow())

if __name__ == "__main__":
    main()
