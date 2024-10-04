# ml_copilot_agent/__init__.py

import os
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings

def initialize(api_key: str, model: str = "gpt-4o", temperature: float = 0.1):
    """Initialize the ML Copilot with API key and LLM settings."""
    os.environ["OPENAI_API_KEY"] = api_key
    Settings.llm = OpenAI(model=model, temperature=temperature)
