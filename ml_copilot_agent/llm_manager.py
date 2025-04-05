# ml_copilot_agent/llm_manager.py

import os
import logging
from typing import Tuple, Optional

# LlamaIndex LLM imports
from llama_index.llms.openai import OpenAI
from llama_index.llms.ollama import Ollama
from llama_index.core.base.llms.base import BaseLLM
from llama_index.core.base.embeddings.base import BaseEmbedding

# LlamaIndex Embedding imports
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Configuration
from .config import (
    DEFAULT_LLM_PROVIDER, DEFAULT_OPENAI_MODEL, DEFAULT_OLLAMA_MODEL,
    DEFAULT_OLLAMA_BASE_URL, OLLAMA_REQUEST_TIMEOUT,
    DEFAULT_OPENAI_EMBEDDING_MODEL, DEFAULT_OLLAMA_EMBEDDING_MODEL
)

logger = logging.getLogger(__name__)

class LLMManager:
    """Handles selection and initialization of LLMs and Embedding models."""

    async def configure_llm(self) -> Tuple[Optional[BaseLLM], Optional[BaseEmbedding]]:
        """
        Interactively guides the user to select and configure the LLM and Embedding model.

        Returns:
            A tuple containing the initialized LLM instance and Embedding instance,
            or (None, None) if configuration fails.
        """
        print("\n--- LLM Configuration ---")
        print(f"Available providers: 1. OpenAI (API, Default: {DEFAULT_OPENAI_MODEL}), 2. Ollama (Local, Default: {DEFAULT_OLLAMA_MODEL})")

        llm_provider = DEFAULT_LLM_PROVIDER
        while True:
            choice = input(f"Select LLM provider (1 or 2, press Enter for default '{DEFAULT_LLM_PROVIDER}'): ").strip()
            if not choice:
                break
            if choice == '1':
                llm_provider = "openai"
                break
            if choice == '2':
                llm_provider = "ollama"
                break
            print("Invalid choice.")

        logger.info(f"Selected LLM provider: {llm_provider}")

        if llm_provider == "openai":
            return self._configure_openai()
        elif llm_provider == "ollama":
            return await self._configure_ollama()
        else:
            logger.error(f"Invalid LLM provider selected: {llm_provider}")
            return None, None

    def _configure_openai(self) -> Tuple[Optional[OpenAI], Optional[OpenAIEmbedding]]:
        """Configures OpenAI LLM and Embedding model."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            api_key = input("Enter your OpenAI API Key: ").strip()
            if not api_key:
                logger.error("OpenAI API Key is required.")
                print("Error: OpenAI API Key cannot be empty.")
                return None, None
            # Optionally save it to env for the session if needed elsewhere, but be careful with security
            # os.environ["OPENAI_API_KEY"] = api_key

        model_name = input(f"Enter OpenAI model name (e.g., gpt-4o, gpt-4o-mini) [default: {DEFAULT_OPENAI_MODEL}]: ").strip() or DEFAULT_OPENAI_MODEL
        embedding_model_name = input(f"Enter OpenAI embedding model name [default: {DEFAULT_OPENAI_EMBEDDING_MODEL}]: ").strip() or DEFAULT_OPENAI_EMBEDDING_MODEL

        try:
            llm = OpenAI(model=model_name, api_key=api_key)
            # Test connection (optional, but good practice)
            # llm.complete("test")
            logger.info(f"OpenAI LLM initialized: {model_name}")

            embed_model = OpenAIEmbedding(model=embedding_model_name, api_key=api_key)
            # Test embedding (optional)
            # embed_model.get_text_embedding("test")
            logger.info(f"OpenAI Embedding model initialized: {embedding_model_name}")

            return llm, embed_model
        except Exception as e:
            logger.exception(f"Failed to initialize OpenAI models: {e}")
            print(f"Error: Failed to initialize OpenAI - {e}")
            return None, None

    async def _configure_ollama(self) -> Tuple[Optional[Ollama], Optional[HuggingFaceEmbedding]]:
        """Configures Ollama LLM and a local HuggingFace Embedding model."""
        base_url = input(f"Enter Ollama base URL [default: {DEFAULT_OLLAMA_BASE_URL}]: ").strip() or DEFAULT_OLLAMA_BASE_URL
        model_name = input(f"Enter Ollama model name (e.g., llama3.1, gemma:7b) [default: {DEFAULT_OLLAMA_MODEL}]: ").strip() or DEFAULT_OLLAMA_MODEL
        embedding_model_name = input(f"Enter HuggingFace embedding model name [default: {DEFAULT_OLLAMA_EMBEDDING_MODEL}]: ").strip() or DEFAULT_OLLAMA_EMBEDDING_MODEL

        try:
            print(f"Attempting to connect to Ollama at {base_url} with model {model_name}...")
            # Check if Ollama server is running and the model exists
            import ollama as ollama_client
            try:
                # List local models to see if the chosen one exists
                local_models = ollama_client.list()['models']
                model_exists = any(m['name'] == model_name for m in local_models)
                if not model_exists:
                    pull_choice = input(f"Model '{model_name}' not found locally. Attempt to pull it? (y/n): ").lower()
                    if pull_choice == 'y':
                        print(f"Pulling '{model_name}' from Ollama Hub (this may take time)...")
                        ollama_client.pull(model_name)
                        print("Model pulled successfully.")
                    else:
                        print("Model not available. Please ensure the model is pulled or choose another.")
                        return None, None
                else:
                     print(f"Model '{model_name}' found locally.")

            except Exception as client_err:
                 print(f"Error communicating with Ollama server at {base_url}: {client_err}")
                 print("Please ensure the Ollama server is running and accessible.")
                 return None, None


            llm = Ollama(
                model=model_name,
                base_url=base_url,
                request_timeout=OLLAMA_REQUEST_TIMEOUT
            )
            # Test connection (optional, but good practice)
            # await llm.acomplete("test") # Use async completion
            logger.info(f"Ollama LLM initialized: {model_name} at {base_url}")

            # Initialize local embedding model
            # Note: This might download the model files on first use
            print(f"Initializing HuggingFace embedding model: {embedding_model_name}...")
            embed_model = HuggingFaceEmbedding(model_name=embedding_model_name)
            # Test embedding (optional)
            # embed_model.get_text_embedding("test")
            logger.info(f"HuggingFace Embedding model initialized: {embedding_model_name}")
            print("Embedding model initialized.")

            return llm, embed_model
        except Exception as e:
            logger.exception(f"Failed to initialize Ollama/HF models: {e}")
            print(f"Error: Failed to initialize Ollama/HF models - {e}")
            return None, None
