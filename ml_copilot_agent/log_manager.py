# ml_copilot_agent/log_manager.py

import os
import json
import logging
import datetime
from typing import Optional, Dict, Any

from .config import LOG_FILENAME, LOG_LEVEL, LOG_FORMAT, LOG_DATE_FORMAT

# --- Root Logger Configuration ---
# This should be called once at the application entry point (__main__.py)
def configure_logging(level=LOG_LEVEL, log_format=LOG_FORMAT, date_format=LOG_DATE_FORMAT):
    """Configures the root logger."""
    logging.basicConfig(level=level, format=log_format, datefmt=date_format)
    # You might want to add handlers here, e.g., a StreamHandler for console output
    # logging.getLogger().addHandler(logging.StreamHandler()) # Already added by basicConfig
    logging.getLogger("httpx").setLevel(logging.WARNING) # Reduce verbosity of http library
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("ollama").setLevel(logging.WARNING)


# --- Project-Specific JSON Logger ---
class LogManager:
    """Handles writing structured logs to a project-specific JSON Lines file."""

    def __init__(self, project_path: str):
        """
        Initializes the LogManager for a specific project.

        Args:
            project_path: The root directory of the current project.
        """
        self.log_file_path = os.path.join(project_path, LOG_FILENAME)
        self._ensure_log_file_exists()
        # Get a standard logger instance for internal messages if needed
        self.internal_logger = logging.getLogger(f"{__name__}.{os.path.basename(project_path)}")
        self.internal_logger.info(f"Log manager initialized. Logging to: {self.log_file_path}")

    def _ensure_log_file_exists(self):
        """Creates the log file and its directory if they don't exist."""
        try:
            os.makedirs(os.path.dirname(self.log_file_path), exist_ok=True)
            # Create file if it doesn't exist, do nothing if it does
            with open(self.log_file_path, 'a') as f:
                pass
        except OSError as e:
            # Use standard logging here as the file path might be the issue
            logging.error(f"Failed to ensure log file exists at {self.log_file_path}: {e}")
            # Potentially raise the error or handle it depending on required robustness

    def log(self, level: str, message: str, event: Optional[str] = None,
            data: Optional[Dict[str, Any]] = None, exc_info: bool = False):
        """
        Writes a structured log entry to the project's JSON Lines file.

        Args:
            level: Log level (e.g., "INFO", "WARN", "ERROR", "DEBUG", "ACTION", "CODE_EXEC").
            message: The main log message.
            event: The name of the workflow event/step (optional).
            data: Additional structured data related to the log entry (optional).
            exc_info: If True, includes exception traceback information (use in except blocks).
        """
        log_entry = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "level": level.upper(),
            "event": event,
            "message": message,
            "data": data if data is not None else {},
        }

        if exc_info:
            # Capture traceback if requested
            import traceback
            log_entry["traceback"] = traceback.format_exc()

        try:
            with open(self.log_file_path, 'a') as f:
                json.dump(log_entry, f)
                f.write('\n') # Write each entry as a new line (JSON Lines format)
        except (OSError, TypeError) as e:
            # Fallback to standard logging if writing to file fails
            self.internal_logger.error(f"Failed to write log entry to {self.log_file_path}: {e}")
            self.internal_logger.error(f"Log Entry Data: {log_entry}")

