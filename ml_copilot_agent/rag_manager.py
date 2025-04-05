# ml_copilot_agent/rag_manager.py

import os
import logging
import json
from typing import List, Optional

# LlamaIndex imports
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.base.llms.base import BaseLLM
import chromadb # Import chromadb client library

# Configuration and Managers
from .config import (RAG_DATA_DIR, RAG_VECTOR_STORE_DIR, RAG_LOG_INDEX_DIR,
                     LOG_FILENAME, RAG_SIMILARITY_TOP_K, RAG_CHUNK_SIZE, RAG_CHUNK_OVERLAP)
from .log_manager import LogManager

logger = logging.getLogger(__name__)

class RAGManager:
    """Handles RAG setup, indexing, and querying for logs and learned documents."""

    def __init__(self, project_path: str, embed_model: BaseEmbedding, llm: BaseLLM, log_manager: LogManager):
        """
        Initializes the RAGManager.

        Args:
            project_path: Path to the current project directory.
            embed_model: The embedding model instance to use.
            llm: The LLM instance (needed for query engine).
            log_manager: The LogManager instance for logging.
        """
        self.project_path = project_path
        self.embed_model = embed_model
        self.llm = llm
        self.log_manager = log_manager

        self.rag_base_dir = os.path.join(project_path, RAG_DATA_DIR)
        self.learn_store_path = os.path.join(self.rag_base_dir, RAG_VECTOR_STORE_DIR)
        self.log_store_path = os.path.join(self.rag_base_dir, RAG_LOG_INDEX_DIR)
        self.log_file_path = os.path.join(project_path, LOG_FILENAME)

        os.makedirs(self.learn_store_path, exist_ok=True)
        os.makedirs(self.log_store_path, exist_ok=True)

        self.learn_index = None
        self.log_index = None

        logger.info("RAG Manager Initialized.")
        logger.info(f"Learn Vector Store Path: {self.learn_store_path}")
        logger.info(f"Log Vector Store Path: {self.log_store_path}")

    # --- Document Loading and Indexing (for 'Learn Something New') ---

    def _load_documents(self, file_paths: List[str]) -> List:
        """Loads documents from specified file paths."""
        # Ensure paths are within the project or a designated safe area if needed
        # For simplicity, assuming paths are valid for now
        documents = []
        for file_path in file_paths:
             if not os.path.exists(file_path):
                 logger.warning(f"File not found during RAG loading: {file_path}. Skipping.")
                 continue
             try:
                 # SimpleDirectoryReader handles various file types
                 reader = SimpleDirectoryReader(input_files=[file_path])
                 docs = reader.load_data()
                 # Add file path metadata
                 for doc in docs:
                     doc.metadata["file_path"] = file_path
                 documents.extend(docs)
                 logger.info(f"Loaded {len(docs)} document sections from: {file_path}")
             except Exception as e:
                 logger.error(f"Failed to load document {file_path}: {e}")
                 self.log_manager.log("ERROR", f"Failed to load document for RAG: {file_path}", data={"error": str(e)})
        return documents

    async def setup_learning_rag(self, file_paths: List[str]) -> str:
        """Loads, indexes, and stores documents for the 'Learn' RAG."""
        self.log_manager.log("ACTION", "Setting up Learning RAG", data={"files": file_paths})
        try:
            # 1. Load Documents
            documents = self._load_documents(file_paths)
            if not documents:
                return "Error: No valid documents could be loaded."

            # 2. Initialize ChromaDB client and collection for learning
            db_learn = chromadb.PersistentClient(path=self.learn_store_path)
            chroma_collection_learn = db_learn.get_or_create_collection("learning_docs")
            vector_store_learn = ChromaVectorStore(chroma_collection=chroma_collection_learn)
            storage_context_learn = StorageContext.from_defaults(vector_store=vector_store_learn)

            # 3. Create or Update Index
            # Define node parser/text splitter
            node_parser = SentenceSplitter(chunk_size=RAG_CHUNK_SIZE, chunk_overlap=RAG_CHUNK_OVERLAP)

            # Build the index
            self.learn_index = VectorStoreIndex.from_documents(
                documents,
                storage_context=storage_context_learn,
                embed_model=self.embed_model,
                transformations=[node_parser],
                show_progress=True # Show progress during indexing
            )
            # Persistence is handled by ChromaDB PersistentClient in this case

            logger.info(f"Learning RAG index created/updated at {self.learn_store_path}")
            self.log_manager.log("SUCCESS", "Learning RAG setup complete", data={"num_docs_processed": len(documents), "index_path": self.learn_store_path})
            # Update state via memory manager would happen in the workflow step calling this
            return f"Successfully set up RAG for {len(documents)} document sections from {len(file_paths)} files. You can now query them."

        except Exception as e:
            logger.exception("Error during Learning RAG setup")
            self.log_manager.log("ERROR", "Learning RAG setup failed", data={"error": str(e)}, exc_info=True)
            return f"Error setting up Learning RAG: {e}"

    async def query_learn_index(self, query: str) -> str:
        """Queries the 'Learn' RAG index."""
        self.log_manager.log("ACTION", "Querying Learning RAG", data={"query": query})
        if self.learn_index is None:
            # Attempt to load existing index if not in memory
            logger.info("Learning index not in memory, attempting to load from storage...")
            try:
                db_learn = chromadb.PersistentClient(path=self.learn_store_path)
                chroma_collection_learn = db_learn.get_or_create_collection("learning_docs") # Use get_or_create
                vector_store_learn = ChromaVectorStore(chroma_collection=chroma_collection_learn)
                self.learn_index = VectorStoreIndex.from_vector_store(
                    vector_store=vector_store_learn,
                    embed_model=self.embed_model,
                )
                logger.info("Successfully loaded Learning index from storage.")
            except Exception as e:
                logger.error(f"Failed to load Learning index from {self.learn_store_path}: {e}")
                self.log_manager.log("ERROR", "Failed to load Learning RAG index", data={"error": str(e)})
                return "Error: Learning RAG index is not available or couldn't be loaded. Please set it up first."

        try:
            query_engine = self.learn_index.as_query_engine(
                similarity_top_k=RAG_SIMILARITY_TOP_K,
                llm=self.llm
            )
            response = await query_engine.aquery(query)
            response_text = str(response) # Extract text from response object
            self.log_manager.log("SUCCESS", "Learning RAG query successful", data={"query": query, "response_snippet": response_text[:200]})
            return response_text
        except Exception as e:
            logger.exception("Error during Learning RAG query")
            self.log_manager.log("ERROR", "Learning RAG query failed", data={"query": query, "error": str(e)}, exc_info=True)
            return f"Error querying learned documents: {e}"

    # --- Log Loading and Indexing ---

    def _load_log_entries(self) -> List:
        """Loads log entries from the JSON Lines file."""
        from llama_index.core import Document # Local import
        log_documents = []
        try:
            with open(self.log_file_path, 'r') as f:
                for i, line in enumerate(f):
                    try:
                        entry = json.loads(line)
                        # Create a text representation for indexing
                        text_content = f"Timestamp: {entry.get('timestamp', 'N/A')}\nLevel: {entry.get('level', 'N/A')}\nEvent: {entry.get('event', 'N/A')}\nMessage: {entry.get('message', '')}"
                        if entry.get('data'):
                            text_content += f"\nData: {json.dumps(entry['data'])}" # Include data
                        # Create a Document object
                        doc = Document(
                            text=text_content,
                            metadata={
                                "log_level": entry.get('level', 'UNKNOWN'),
                                "event": entry.get('event', 'NONE'),
                                "timestamp": entry.get('timestamp', ''),
                                "line_number": i + 1
                            }
                        )
                        log_documents.append(doc)
                    except json.JSONDecodeError:
                        logger.warning(f"Skipping invalid JSON line {i+1} in log file: {self.log_file_path}")
                        continue
            logger.info(f"Loaded {len(log_documents)} log entries from {self.log_file_path}")
            return log_documents
        except OSError as e:
            logger.error(f"Error reading log file {self.log_file_path}: {e}")
            self.log_manager.log("ERROR", f"Failed to read log file for RAG: {self.log_file_path}", data={"error": str(e)})
            return []
        except Exception as e:
             logger.exception(f"Unexpected error loading log entries from {self.log_file_path}")
             self.log_manager.log("ERROR", f"Unexpected error loading log file for RAG: {self.log_file_path}", data={"error": str(e)})
             return []

    async def ensure_log_index(self, force_reindex=False) -> bool:
        """Ensures the log index exists, creating or updating it if necessary."""
        self.log_manager.log("ACTION", "Ensuring Log RAG index exists", data={"force_reindex": force_reindex})
        try:
             # Initialize ChromaDB client and collection for logs
            db_log = chromadb.PersistentClient(path=self.log_store_path)
            chroma_collection_log = db_log.get_or_create_collection("project_logs")
            vector_store_log = ChromaVectorStore(chroma_collection=chroma_collection_log)
            storage_context_log = StorageContext.from_defaults(vector_store=vector_store_log)

            # Check if index needs building/rebuilding
            # Simple check: Re-index if forced or if log file modified time > index modified time (tricky with vector stores)
            # Easier approach: Always reload/rebuild for logs as they change frequently.
            # Or, check if the collection is empty?
            should_reindex = force_reindex or chroma_collection_log.count() == 0 # Reindex if empty or forced

            if not should_reindex and self.log_index:
                 logger.info("Log index already in memory and no reindex forced.")
                 return True

            if not should_reindex and not self.log_index:
                 # Try loading if not forced and not in memory
                 try:
                      self.log_index = VectorStoreIndex.from_vector_store(
                           vector_store=vector_store_log,
                           embed_model=self.embed_model,
                      )
                      logger.info("Successfully loaded existing Log index from storage.")
                      return True
                 except Exception as load_err:
                      logger.warning(f"Could not load existing log index (will rebuild): {load_err}")
                      should_reindex = True # Force reindex if loading failed

            if should_reindex:
                logger.info("Building/Rebuilding Log RAG index...")
                log_documents = self._load_log_entries()
                if not log_documents:
                    logger.warning("No log entries found to index.")
                    # Clear existing collection if rebuilding? Or just don't index?
                    # Let's just not index if no docs.
                    self.log_index = None # Ensure index is None if empty
                    return False # Indicate index is not ready

                # Clear existing collection before re-indexing (optional, ensures freshness)
                # Be careful with this in production. For simplicity, let's just add/update.
                # db_log.delete_collection("project_logs")
                # chroma_collection_log = db_log.create_collection("project_logs")
                # vector_store_log = ChromaVectorStore(chroma_collection=chroma_collection_log)
                # storage_context_log = StorageContext.from_defaults(vector_store=vector_store_log)

                node_parser = SentenceSplitter(chunk_size=RAG_CHUNK_SIZE, chunk_overlap=RAG_CHUNK_OVERLAP)
                self.log_index = VectorStoreIndex.from_documents(
                    log_documents,
                    storage_context=storage_context_log,
                    embed_model=self.embed_model,
                    transformations=[node_parser],
                    show_progress=True
                )
                logger.info(f"Log RAG index created/updated at {self.log_store_path}")
                self.log_manager.log("SUCCESS", "Log RAG index build/update complete", data={"num_logs_processed": len(log_documents), "index_path": self.log_store_path})
                # Update state via memory manager would happen in the workflow step calling this
                return True
            else:
                 # This case should ideally be covered by the loading logic above
                 logger.info("Log index exists and no reindex needed.")
                 return True

        except Exception as e:
            logger.exception("Error during Log RAG index ensure/build")
            self.log_manager.log("ERROR", "Log RAG index ensure/build failed", data={"error": str(e)}, exc_info=True)
            self.log_index = None
            return False

    async def query_log_index(self, query: str) -> str:
        """Queries the log RAG index."""
        self.log_manager.log("ACTION", "Querying Log RAG", data={"query": query})

        # Ensure index is ready before querying
        index_ready = await self.ensure_log_index(force_reindex=False) # Don't force reindex on every query
        if not index_ready or self.log_index is None:
             return "Error: Log RAG index is not available. Please try again or check logs."

        try:
            query_engine = self.log_index.as_query_engine(
                similarity_top_k=RAG_SIMILARITY_TOP_K,
                llm=self.llm
            )
            response = await query_engine.aquery(query)
            response_text = str(response)
            self.log_manager.log("SUCCESS", "Log RAG query successful", data={"query": query, "response_snippet": response_text[:200]})
            return response_text
        except Exception as e:
            logger.exception("Error during Log RAG query")
            self.log_manager.log("ERROR", "Log RAG query failed", data={"query": query, "error": str(e)}, exc_info=True)
            return f"Error querying logs: {e}"