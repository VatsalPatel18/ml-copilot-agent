# ml_copilot_agent/project_manager.py

import os
import re
import logging

logger = logging.getLogger(__name__)

class ProjectManager:
    """Handles creation, selection, and path management for projects."""

    def __init__(self, projects_base_dir: str):
        """
        Initializes the ProjectManager.

        Args:
            projects_base_dir: The root directory where all projects are stored.
        """
        self.projects_base_dir = projects_base_dir
        os.makedirs(self.projects_base_dir, exist_ok=True)
        logger.info(f"Project base directory: {self.projects_base_dir}")

    def list_projects(self) -> list[str]:
        """Lists existing projects."""
        try:
            return [d for d in os.listdir(self.projects_base_dir)
                    if os.path.isdir(os.path.join(self.projects_base_dir, d))]
        except OSError as e:
            logger.error(f"Error listing projects in {self.projects_base_dir}: {e}")
            return []

    def _sanitize_project_name(self, name: str) -> str:
        """Sanitizes a project name to be filesystem-friendly."""
        name = re.sub(r'[^\w\- ]', '', name) # Allow letters, numbers, underscore, hyphen, space
        name = re.sub(r'\s+', '_', name) # Replace spaces with underscores
        return name.strip('_')[:50] # Limit length and remove leading/trailing underscores

    def create_project(self, project_name: str) -> str | None:
        """Creates a new project directory."""
        sanitized_name = self._sanitize_project_name(project_name)
        if not sanitized_name:
            logger.error("Invalid project name after sanitization.")
            print("Error: Invalid project name provided.")
            return None

        project_path = os.path.join(self.projects_base_dir, sanitized_name)
        if os.path.exists(project_path):
            logger.warning(f"Project '{sanitized_name}' already exists.")
            print(f"Project '{sanitized_name}' already exists.")
            return sanitized_name # Return existing name

        try:
            os.makedirs(project_path)
            # Optionally create subdirs like data, models, results here if needed immediately
            os.makedirs(os.path.join(project_path, "data"), exist_ok=True)
            os.makedirs(os.path.join(project_path, "models"), exist_ok=True)
            os.makedirs(os.path.join(project_path, "results"), exist_ok=True)
            os.makedirs(os.path.join(project_path, "plots"), exist_ok=True)
            os.makedirs(os.path.join(project_path, "rag_data"), exist_ok=True)
            os.makedirs(os.path.join(project_path, "rag_data", "vector_store"), exist_ok=True)
            os.makedirs(os.path.join(project_path, "rag_data", "log_index"), exist_ok=True)

            logger.info(f"Created new project: {sanitized_name} at {project_path}")
            print(f"Created new project: '{sanitized_name}'")
            return sanitized_name
        except OSError as e:
            logger.error(f"Error creating project directory {project_path}: {e}")
            print(f"Error: Could not create project '{sanitized_name}'.")
            return None

    def get_project_path(self, project_name: str) -> str:
        """Gets the full path for a given project name."""
        # Assume project_name is already sanitized if coming from list/create
        return os.path.join(self.projects_base_dir, project_name)

    def select_or_create_project(self) -> str | None:
        """Interactively guides the user to select or create a project."""
        print("\n--- Project Selection ---")
        existing_projects = self.list_projects()

        if existing_projects:
            print("Existing projects:")
            for i, name in enumerate(existing_projects):
                print(f"  {i + 1}. {name}")
            print(f"  {len(existing_projects) + 1}. Create a new project")

            while True:
                try:
                    choice = input(f"Select a project number (1-{len(existing_projects) + 1}): ")
                    choice_num = int(choice)
                    if 1 <= choice_num <= len(existing_projects):
                        selected_project = existing_projects[choice_num - 1]
                        logger.info(f"User selected existing project: {selected_project}")
                        return selected_project
                    elif choice_num == len(existing_projects) + 1:
                        break # Proceed to create new project
                    else:
                        print("Invalid choice. Please enter a valid number.")
                except ValueError:
                    print("Invalid input. Please enter a number.")
        else:
            print("No existing projects found.")

        # Create new project
        while True:
            new_project_name = input("Enter a name for the new project: ").strip()
            if new_project_name:
                created_name = self.create_project(new_project_name)
                if created_name:
                    return created_name
                # If creation failed, loop again
            else:
                print("Project name cannot be empty.")
