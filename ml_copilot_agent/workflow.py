# ml_copilot_agent/workflow.py

import os
import asyncio
import json
from typing import Optional, Union, List, Dict, Any
import traceback # Import traceback for better error logging

# Core workflow components
from llama_index.core.workflow import (
    Workflow, Context, Event, StartEvent, StopEvent, step
)
# Removed: from llama_index.utils.workflow import WorkflowCheckpointer

from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.tools.code_interpreter.base import CodeInterpreterToolSpec
from llama_index.agent.openai import OpenAIAgent
from llama_index.core import Settings
from llama_index.core.utils import get_tokenizer # Import tokenizer utility

# ===========================================================
# 1. Simplified Event Definitions (Keep as before)
# ===========================================================

class UserInputEvent(Event):
    """Event carrying the raw user input."""
    user_input: str

class ExecuteCodeEvent(Event):
    """Event carrying the prompt to be executed by the code interpreter."""
    prompt: str
    task_description: str # e.g., "Preprocessing", "Training", "Plotting"

class AskUserEvent(Event):
    """Event indicating the agent needs more information from the user."""
    question: str

class ListFilesEvent(Event):
    """Event to trigger listing files."""
    pass

class AgentResponseEvent(Event):
    """Event carrying a direct text response from the agent (not code execution)."""
    response: str

# ===========================================================
# 2. Enhanced MLWorkflow Definition
# ===========================================================

class MLWorkflow(Workflow):
    def __init__(self, memory: ChatMemoryBuffer, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # --- Agent Setup ---
        if not Settings.llm:
            raise ValueError("LLM not configured in Settings. Please set Settings.llm.")
        self.llm = Settings.llm # LLM for interpreting commands

        # Code Interpreter Agent (ensure API key is set via initialize)
        code_spec = CodeInterpreterToolSpec()
        tools = code_spec.to_tool_list()
        # Ensure the code agent also uses the configured LLM
        self.code_agent = OpenAIAgent.from_tools(tools, llm=self.llm, verbose=True)

        # --- Memory ---
        self.memory = memory

        # --- Internal State Tracking (Example - can be expanded) ---
        # This helps the interpreter LLM know what's available
        self.internal_state = {
            "available_data_vars": [],
            "available_model_paths": [],
            "last_result_path": None,
            "current_task": None,
        }
        # Attempt to get a tokenizer for memory management
        try:
            self.tokenizer = get_tokenizer()
            print("Tokenizer loaded for memory management.")
        except ImportError:
            print("Warning: Default tokenizer not found. Using basic length check for memory.")
            self.tokenizer = len # Fallback to simple length
        self.memory.tokenizer_fn = self.tokenizer # Assign tokenizer to memory

    # --- Helper Methods ---
    def _update_internal_state(self, updates: Dict[str, Any]):
        """Safely update the internal state dictionary."""
        for key, value in updates.items():
            if key in self.internal_state:
                if isinstance(self.internal_state[key], list) and isinstance(value, list):
                    # Append unique items for lists
                    self.internal_state[key].extend([item for item in value if item not in self.internal_state[key]])
                elif isinstance(self.internal_state[key], list) and not isinstance(value, list):
                     if value not in self.internal_state[key]:
                         self.internal_state[key].append(value)
                else:
                    # Overwrite for non-list types or if types mismatch
                    self.internal_state[key] = value
            else:
                # Add new key if it doesn't exist
                self.internal_state[key] = value
        print(f"[State Update]: {updates}") # Log state changes
        print(f"[Current State]: {json.dumps(self.internal_state, indent=2)}") # Pretty print state


    def _get_current_context_summary(self) -> str:
        """Generates a summary of the current state for the LLM."""
        summary_lines = ["Current Context Summary:"]
        if self.internal_state["available_data_vars"]:
            summary_lines.append(f"- Available data variables: {', '.join(f'`{v}`' for v in self.internal_state['available_data_vars'])}") # Added backticks
        if self.internal_state["available_model_paths"]:
            summary_lines.append(f"- Available model paths: {', '.join(self.internal_state['available_model_paths'])}")
        if self.internal_state["last_result_path"]:
            summary_lines.append(f"- Last saved result path: {self.internal_state['last_result_path']}")
        if self.internal_state["current_task"]:
             summary_lines.append(f"- Currently working on: {self.internal_state['current_task']}")
        if len(summary_lines) == 1: # Only the header
            summary_lines.append("- No data or models loaded yet.")
        return "\n".join(summary_lines)

    async def _interpret_command(self, user_input: str) -> Union[ExecuteCodeEvent, AskUserEvent, ListFilesEvent, StopEvent, AgentResponseEvent]:
        """Uses LLM to interpret user input based on history and context."""
        print("\n--- Interpreting User Input ---")
        # Get limited history to avoid overly long prompts
        # Calculate token count for current input to pass to memory.get
        current_input_tokens = len(self.tokenizer(user_input))
        history = self.memory.get(initial_token_count=current_input_tokens)
        context_summary = self._get_current_context_summary()

        # Prepare messages for the interpreter LLM
        system_prompt = (
            "You are an AI assistant interpreting commands for an ML Copilot agent that uses a code interpreter tool. "
            "Your goal is to understand the user's request based on the conversation history and current context, "
            "extract necessary parameters, and decide the next step.\n"
            "Available high-level tasks: 'load data', 'preprocess', 'cluster', 'train', 'evaluate', 'plot', "
            "'analyze survival', 'select model', 'list files', 'custom task'.\n"
            f"{context_summary}\n" # Provide current state
            "Based on the latest user input and the conversation history below, determine the user's primary intent. \n"
            "1. **Execute Code:** If the intent and all necessary parameters (like file paths, variable names (`var_name`), column names, model types, save paths) are clear, "
            "formulate a detailed, unambiguous prompt for the code execution agent. Ensure the prompt specifies input variable names and desired output variable names or save paths. "
            "Respond ONLY with a valid JSON object: "
            '`{"action": "execute", "task_description": "Brief Task Name", "prompt": "Detailed prompt for code agent..."}`.\n'
            "2. **Ask User:** If the intent is clear but parameters are missing or ambiguous (e.g., needs file path, column name, variable name), ask a specific clarifying question. "
            "Respond ONLY with a valid JSON object: "
            '`{"action": "ask", "question": "Your specific question..."}`.\n'
            "3. **List Files:** If the user explicitly asks to list files, respond ONLY with: "
            '`{"action": "list_files"}`.\n'
            "4. **Exit:** If the user clearly wants to exit or stop, respond ONLY with: "
            '`{"action": "exit"}`.\n'
            "5. **Respond:** If the user asks a general question (e.g., 'what can you do?', 'help') or makes a comment not requiring code execution, provide a helpful text response. "
            "Respond ONLY with a valid JSON object: "
            '`{"action": "respond", "response": "Your helpful text response..."}`.\n'
            "--- IMPORTANT ---:\n"
            "- Respond ONLY with a single, valid JSON object matching one of the formats above.\n"
            "- Do NOT add any introductory text, explanations, or formatting around the JSON.\n"
            "- When generating prompts for the code agent (`action: execute`), be precise about variable names mentioned in the context summary."
        )
        messages = [ChatMessage(role=MessageRole.SYSTEM, content=system_prompt)]
        messages.extend(history) # Add conversation history
        # Add latest user input if it's not already the last message in history
        if not history or history[-1].role != MessageRole.USER or history[-1].content != user_input:
             messages.append(ChatMessage(role=MessageRole.USER, content=user_input))

        try:
            print(f"[Interpreter Input]: History Length={len(history)}, Context Summary Provided.")
            # print(f"[Interpreter Messages]: {[m.dict() for m in messages]}") # DEBUG: See full prompt to interpreter
            response = await self.llm.achat(messages=messages)
            llm_output = response.message.content.strip() # Strip whitespace
            print(f"[Interpreter Raw Output]: {llm_output}")

            # Attempt to parse the LLM response as JSON
            # Handle potential markdown code block ```json ... ```
            if llm_output.startswith("```json"):
                llm_output = llm_output[7:-3].strip() # Remove ```json and ```
            elif llm_output.startswith("```"):
                 llm_output = llm_output[3:-3].strip() # Remove ```

            decision = json.loads(llm_output)
            action = decision.get("action")

            if action == "execute":
                prompt = decision.get("prompt")
                task_desc = decision.get("task_description", "Executing Task")
                if prompt:
                    # Update internal state about the task being started
                    self._update_internal_state({"current_task": task_desc})
                    return ExecuteCodeEvent(prompt=prompt, task_description=task_desc)
                else:
                     # If LLM fails, ask user to be more specific
                     print("[Interpreter Fallback]: LLM decided to execute but provided no prompt.")
                     return AskUserEvent(question="I understood you want to execute a task, but I couldn't formulate the exact code steps. Could you please provide more specific details or parameters?")
            elif action == "ask":
                question = decision.get("question")
                if question:
                    return AskUserEvent(question=question)
                else:
                    # If LLM fails, ask user to clarify generally
                    print("[Interpreter Fallback]: LLM decided to ask but provided no question.")
                    return AskUserEvent(question="I need a bit more information to proceed. Could you please clarify your request?")
            elif action == "list_files":
                return ListFilesEvent()
            elif action == "exit":
                return StopEvent(result="Workflow terminated by user request.")
            elif action == "respond":
                 response_text = decision.get("response")
                 if response_text:
                     return AgentResponseEvent(response=response_text)
                 else:
                     # If LLM fails, provide generic response
                     print("[Interpreter Fallback]: LLM decided to respond but provided no text.")
                     return AgentResponseEvent(response="Okay, I acknowledge that.") # Generic fallback
            else:
                 # If action is unknown
                 print(f"[Interpreter Fallback]: LLM returned unknown action: {action}")
                 return AskUserEvent(question="I'm not sure how to handle that request. Could you please rephrase?")

        except json.JSONDecodeError:
            print(f"[Interpreter Error]: Failed to parse LLM response as JSON. Raw response: {llm_output}")
            # Fallback: Ask the user to clarify or try again, include LLM output for context
            return AskUserEvent(question=f"I had trouble understanding that response structure. Could you please rephrase or clarify your request? (Internal response: {llm_output[:100]}...)")
        except Exception as e:
            print(f"[Interpreter Error]: An unexpected error occurred: {e}")
            traceback.print_exc() # Print full traceback for debugging
            return AskUserEvent(question="Sorry, an internal error occurred while interpreting your command. Please try again.")


    # --- Workflow Steps ---

    # Removed step_id="start_conversation"
    @step
    async def start_conversation(self, ctx: Context, ev: StartEvent) -> UserInputEvent:
        """Initializes the workflow and gets the first user command."""
        print("\n--- ML Copilot Initialized ---")
        initial_message = ("Hello! I am your ML Copilot. I can help you with tasks like loading data, "
                           "preprocessing, training models, evaluation, plotting, and more using code execution. "
                           "How can I assist you today? (Type 'help' for more options, 'exit' to quit)")
        print(f"Agent: {initial_message}") # Make it clear it's the agent speaking
        self.memory.put(ChatMessage(role=MessageRole.ASSISTANT, content=initial_message))
        user_input = input("You: ").strip() # Make it clear user is typing
        self.memory.put(ChatMessage(role=MessageRole.USER, content=user_input))
        return UserInputEvent(user_input=user_input)

    # Removed step_id="interpret_user_input"
    @step
    async def interpret_user_input(self, ctx: Context, ev: UserInputEvent) -> Union[ExecuteCodeEvent, AskUserEvent, ListFilesEvent, StopEvent, AgentResponseEvent]:
        """Interprets the user's input using an LLM."""
        return await self._interpret_command(ev.user_input)

    # Removed step_id="execute_code"
    @step
    async def execute_code(self, ctx: Context, ev: ExecuteCodeEvent) -> UserInputEvent:
        """Executes the code prompt using the code agent."""
        print(f"\n--- Executing Code: {ev.task_description} ---")
        # print(f"Prompt for Code Agent:\n```\n{ev.prompt}\n```") # Log the prompt being sent (optional)
        follow_up_message = f"An internal error occurred during {ev.task_description}. Please check logs. What would you like to do?" # Default error message
        try:
            # Use the code agent to generate and execute code asynchronously
            # Ensure the agent uses the correct LLM settings implicitly handled by OpenAIAgent
            response = await asyncio.get_event_loop().run_in_executor(
                None, self.code_agent.chat, ev.prompt
            )
            agent_response_content = str(response) # Agent's response includes execution output
            print("\n--- Code Agent Response ---")
            print(agent_response_content)
            print("-------------------------\n")

            # Add agent's execution output/response to memory
            # Truncate long outputs to avoid exceeding memory limits quickly
            max_output_len = 2000 # Example limit
            truncated_response = agent_response_content
            if len(truncated_response) > max_output_len:
                truncated_response = agent_response_content[:max_output_len] + "\n... [Output Truncated]"
            self.memory.put(ChatMessage(role=MessageRole.ASSISTANT, content=f"[Code Execution Output for '{ev.task_description}']\n{truncated_response}"))


            # --- Attempt to extract state updates from the prompt/response ---
            potential_updates = {}
            lines = ev.prompt.split('\n')
            for line in lines:
                 if "save" in line.lower() and "to" in line.lower():
                      parts = line.split()
                      try:
                           save_idx = parts.index("to")
                           path = parts[save_idx + 1].strip("'\",`")
                           if "/" in path or "." in path.split('/')[-1]:
                               potential_updates["last_result_path"] = path
                               if path.endswith(".pkl") and "model" in line.lower():
                                   potential_updates["available_model_paths"] = [path]
                      except (ValueError, IndexError): pass
                 if "variable named" in line.lower() or "assign to" in line.lower():
                      parts = line.split()
                      try:
                            var_name = None
                            for i, part in enumerate(parts):
                                if part.strip("`'\",.").isidentifier():
                                     if i > 0 and parts[i-1].lower() in ["named", "to"]:
                                          var_name = part.strip("`'\",.")
                                          break
                            if var_name and (any(kw in line.lower() for kw in ["data", "result"]) or any(prefix in var_name for prefix in ["df", "X_", "y_"])):
                                potential_updates["available_data_vars"] = [var_name]
                      except (ValueError, IndexError): pass

            if potential_updates: self._update_internal_state(potential_updates)

            # Check for errors
            if "error" in agent_response_content.lower() or "exception" in agent_response_content.lower() or "traceback" in agent_response_content.lower():
                print(f"Warning: Potential error detected in {ev.task_description} execution.")
                follow_up_message = f"{ev.task_description} execution finished, but there might have been an error (see output above). What would you like to do next?"
            else:
                 follow_up_message = f"{ev.task_description} finished successfully. What would you like to do next?"

        except Exception as e:
            print(f"\n--- Error during Code Execution ({ev.task_description}) ---")
            error_message = f"An exception occurred: {e}"
            print(error_message)
            traceback.print_exc() # Print full traceback for debugging
            print("-----------------------------------------\n")
            self.memory.put(ChatMessage(role=MessageRole.ASSISTANT, content=f"Error during {ev.task_description}: {error_message}"))
            follow_up_message = f"An error occurred during {ev.task_description}. Please check the error message above. What would you like to do?"
        finally:
             self._update_internal_state({"current_task": None}) # Reset current task

        # Prompt user for next action
        print(f"Agent: {follow_up_message}")
        self.memory.put(ChatMessage(role=MessageRole.ASSISTANT, content=follow_up_message))
        user_input = input("You: ").strip()
        self.memory.put(ChatMessage(role=MessageRole.USER, content=user_input))
        return UserInputEvent(user_input=user_input)

    # Removed step_id="ask_user"
    @step
    async def ask_user(self, ctx: Context, ev: AskUserEvent) -> UserInputEvent:
        """Asks the user a clarifying question."""
        print(f"\nAgent: {ev.question}")
        self.memory.put(ChatMessage(role=MessageRole.ASSISTANT, content=ev.question))
        user_input = input("You: ").strip()
        self.memory.put(ChatMessage(role=MessageRole.USER, content=user_input))
        return UserInputEvent(user_input=user_input)

    # Removed step_id="list_files_step"
    @step
    async def list_files_step(self, ctx: Context, ev: ListFilesEvent) -> UserInputEvent:
        """Lists files in the current directory."""
        print("\n--- Files in Current Directory ---")
        response = ""
        try:
            cwd = os.getcwd()
            response += f"Current directory: {cwd}\n"
            files = os.listdir('.')
            response += "Files/Folders:\n" + ("\n".join(f"- {f}" for f in files) if files else "The directory is empty.")
        except Exception as e:
            response += f"Error listing files: {e}"
        print(response)
        print("--------------------------------\n")
        self.memory.put(ChatMessage(role=MessageRole.ASSISTANT, content=response))

        # Prompt user for next action
        next_prompt = "Listed files. What would you like to do next?"
        print(f"Agent: {next_prompt}")
        self.memory.put(ChatMessage(role=MessageRole.ASSISTANT, content=next_prompt))
        user_input = input("You: ").strip()
        self.memory.put(ChatMessage(role=MessageRole.USER, content=user_input))
        return UserInputEvent(user_input=user_input)

    # Removed step_id="agent_response_step"
    @step
    async def agent_response_step(self, ctx: Context, ev: AgentResponseEvent) -> UserInputEvent:
        """Provides a direct text response from the agent."""
        print(f"\nAgent: {ev.response}")
        self.memory.put(ChatMessage(role=MessageRole.ASSISTANT, content=ev.response))

        # Prompt user for next action
        next_prompt = "What would you like to do next?"
        print(f"Agent: {next_prompt}")
        self.memory.put(ChatMessage(role=MessageRole.ASSISTANT, content=next_prompt))
        user_input = input("You: ").strip()
        self.memory.put(ChatMessage(role=MessageRole.USER, content=user_input))
        return UserInputEvent(user_input=user_input)

    # Removed step_id="stop_workflow"
    @step
    async def stop_workflow(self, ctx: Context, ev: StopEvent) -> None:
        """Handles the stop event."""
        print(f"\n--- Workflow Terminated ---")
        final_message = ev.result or "Workflow stopped."
        print(final_message)
        self.memory.put(ChatMessage(role=MessageRole.ASSISTANT, content=final_message))
        print("---------------------------\n")
        # No return value needed for the final step

