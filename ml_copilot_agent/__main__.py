# ml_copilot/__main__.py

import asyncio
import sys

from .workflow import MLWorkflow
from . import initialize

def main():
    if len(sys.argv) < 2:
        print("Usage: python -m ml_copilot <OPENAI_API_KEY>")
        sys.exit(1)
    api_key = sys.argv[1]
    initialize(api_key)

    async def run_workflow():
        workflow = MLWorkflow(timeout=600, verbose=True)
        await workflow.run()
    
    asyncio.run(run_workflow())

if __name__ == "__main__":
    main()
