# ml_copilot_agent/config.py

import os
import logging

# --- Project Settings ---
PROJECTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'projects'))
STATE_FILENAME = "state.json"
LOG_FILENAME = "logs.jsonl"
RAG_DATA_DIR = "rag_data"
RAG_VECTOR_STORE_DIR = "vector_store"
RAG_LOG_INDEX_DIR = "log_index" # Specific dir for log RAG

# --- LLM Configuration ---
# Options: "openai", "ollama"
DEFAULT_LLM_PROVIDER = "openai"
# OpenAI settings
DEFAULT_OPENAI_MODEL = "gpt-4o-mini" # Or "gpt-4o"
DEFAULT_OPENAI_EMBEDDING_MODEL = "text-embedding-3-small" # Or "text-embedding-ada-002"
# Ollama settings
DEFAULT_OLLAMA_MODEL = "llama3.1" # Or "gemma:2b", "gemma:7b" etc. Needs function calling support ideally.
DEFAULT_OLLAMA_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5" # Example local embedding model
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_REQUEST_TIMEOUT = 300.0 # Seconds, increase for large models/slow hardware

# --- Agent/Workflow Settings ---
DEFAULT_AGENT_TIMEOUT = 600 # Timeout for workflow runs (seconds)
DEFAULT_VERBOSE_LEVEL = True
MAX_TOOL_CALL_ATTEMPTS = 3 # Max retries if agent fails tool call selection

# --- Logging ---
LOG_LEVEL = logging.INFO # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# --- RAG Settings ---
RAG_SIMILARITY_TOP_K = 3
RAG_CHUNK_SIZE = 512
RAG_CHUNK_OVERLAP = 20

# --- ML Task Defaults ---
DEFAULT_PREPROCESS_SAVE_DIR = "data"
DEFAULT_PREPROCESS_FILENAME = "preprocessed_data.csv"
DEFAULT_MODEL_SAVE_DIR = "models"
DEFAULT_MODEL_FILENAME = "model.pkl"
DEFAULT_RESULTS_SAVE_DIR = "results"
DEFAULT_EVALUATION_FILENAME = "evaluation_results.json" # Saving as JSON might be better
DEFAULT_PLOT_SAVE_DIR = "plots" # Will be within results dir

# --- Ensure Project Directory Exists ---
os.makedirs(PROJECTS_DIR, exist_ok=True)
