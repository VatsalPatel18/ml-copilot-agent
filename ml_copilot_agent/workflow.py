import os
import asyncio
from typing import Optional
from typing import Union

from llama_index.core.workflow import (
    Workflow,
    Context,
    Event,
    StartEvent,
    StopEvent,
    step
)
from llama_index.tools.code_interpreter.base import CodeInterpreterToolSpec
from llama_index.agent.openai import OpenAIAgent
from llama_index.llms.openai import OpenAI

# If you need Settings or draw_all_possible_flows
from llama_index.core import Settings
from llama_index.utils.workflow import draw_all_possible_flows


# Define Events
class InitializeEvent(Event):
    pass

class ML_Copilot(Event):
    user_input: str

class CustomEvent(Event):
    custom_instruction: str

class ListFilesEvent(Event):
    pass

class PlotEvent(Event):
    plot_type: str  # 'results' or 'data'
    data_file_path: Optional[str] = None
    additional_instructions: Optional[str] = ''

class PreprocessEvent(Event):
    dataset_path: str
    target_column: str
    save_path: Optional[str] = 'data/preprocessed_data.csv'
    additional_instructions: Optional[str] = ''

class TrainEvent(Event):
    model_path: Optional[str] = 'models/model.pkl'
    additional_instructions: Optional[str] = ''

class EvaluateEvent(Event):
    evaluation_save_path: Optional[str] = 'results/evaluation.txt'
    additional_instructions: Optional[str] = ''

class DocumentEvent(Event):
    report_save_path: Optional[str] = 'reports/report.md'

# Define the MLWorkflow
class MLWorkflow(Workflow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Initialize the code interpreter tool
        code_spec = CodeInterpreterToolSpec()
        tools = code_spec.to_tool_list()
        
        # Create the OpenAIAgent with the code interpreter tool
        self.agent = OpenAIAgent.from_tools(tools, verbose=True)
        # Use the LLM from Settings
        self.llm = Settings.llm

    @step
    async def initialize(self, ctx: Context, ev: StartEvent) -> ML_Copilot:
        # Initialize context data
        await ctx.set('files', os.listdir('.'))
        print("Welcome to ML-Copilot! How can I assist you today?")
        print("I can do the following things: 'show files', 'preprocess data', 'train model', 'evaluate model', 'plot results', 'generate report',  'custom instruction', or 'exit'.")
        user_input = input("> ").strip()
        return ML_Copilot(user_input=user_input)

    @step
    async def ml_copilot(self, ctx: Context, ev: ML_Copilot) -> ML_Copilot | ListFilesEvent | PreprocessEvent| TrainEvent | EvaluateEvent | DocumentEvent | PlotEvent | StopEvent :

        user_input = ev.user_input.lower()
        
        # Simple keyword-based parsing
        if "list files" in user_input or "show files" in user_input:
            return ListFilesEvent()
        elif "preprocess" in user_input:
            # Ask for dataset path and target column
            print("Please enter the dataset path:")
            dataset_path = input("> ").strip()
            print("Please enter the target column name:")
            target_column = input("> ").strip()
            print("Please enter the save path for preprocessed data (press Enter for default 'data/preprocessed_data.csv'):")
            save_path = input("> ").strip() or 'data/preprocessed_data.csv'
            print("Any additional preprocessing instructions? (e.g., 'use standard scaler') (press Enter to skip):")
            additional_instructions = input("> ").strip()
            return PreprocessEvent(
                dataset_path=dataset_path,
                target_column=target_column,
                save_path=save_path,
                additional_instructions=additional_instructions
                )
        elif "train" in user_input:
            print("Please enter the save path for the trained model (press Enter for default 'models/model.pkl'):")
            model_path = input("> ").strip() or 'models/model.pkl'
            print("Any additional training instructions? (e.g., 'use SVM classifier') (press Enter to skip):")
            additional_instructions = input("> ").strip()
            return TrainEvent(model_path=model_path, additional_instructions=additional_instructions)
        elif "evaluate" in user_input:
            print("Please enter the save path for the evaluation results (press Enter for default 'results/evaluation.txt'):")
            evaluation_save_path = input("> ").strip() or 'results/evaluation.txt'
            return EvaluateEvent(evaluation_save_path=evaluation_save_path)
        elif "document" in user_input or "report" in user_input:
            return DocumentEvent()
        elif "exit" in user_input or "quit" in user_input:
            return StopEvent(result="Workflow terminated by user.")
        elif "what can you do" in user_input or "help" in user_input:
            print("I can assist you with the following tasks:")
            print("- 'list files' to show files in the current directory")
            print("- 'preprocess data' to preprocess data for a binary classification task")
            print("- 'train model' to train a binary classification model")
            print("- 'evaluate model' to evaluate the trained model")
            print("- 'generate report' to generate a documentation report")
            print("- 'exit' to terminate the workflow")
            user_input = input("> ").strip()
            return ML_Copilot(user_input=user_input)
        
        elif "plot" in user_input:
            print("What would you like to plot?")
            print("- Type 'data' to plot preprocessed data.")
            print("- Type 'results' to plot evaluation results.")

            plot_type = input("> ").strip().lower()
            if plot_type == 'results':
                return PlotEvent(plot_type='results')
            elif plot_type == 'data':
                return PlotEvent(plot_type='data')
            else:
                print("Invalid option.")
                user_input = input("> ").strip()
                return ML_Copilot(user_input=user_input)
            
        elif "custom" in user_input or "instruction" in user_input:
            print("Please enter your custom instruction:")
            custom_instruction = input("> ").strip()
            return CustomEvent(custom_instruction=custom_instruction)

        else:
            print("I'm sorry, I didn't understand that command.")
            print("You can ask me to 'Show files', 'Preprocess Data', 'Train Model', 'Evaluate Model', 'Plot Results', 'Generate Report', 'Custom Instructions' or 'exit'.")
            user_input = input("> ").strip()
            return ML_Copilot(user_input=user_input)
        
    @step
    async def custom_instruction(self, ctx: Context, ev: CustomEvent) -> Union[CustomEvent, ML_Copilot, StopEvent]:
        custom_instruction = ev.custom_instruction
        print(f"Executing your custom instruction: {custom_instruction}")
        
        prompt = f"""
Please execute the following instruction:

{custom_instruction}
"""
        
        # Use the agent to generate and execute code asynchronously
        response = await asyncio.get_event_loop().run_in_executor(
            None, self.agent.chat, prompt
        )
        
        # Print the agent's response
        print(response)
        
        print("Do you want to enter another set of custom instruction? (yes/no)")
        user_response = input("> ").strip().lower()
        if user_response in ('yes', 'y'):
            print("Please enter your custom instruction:")
            user_input = input("> ").strip()
            return CustomEvent(custom_instruction=user_input)
        else:
            print("What would you like to do next?")
            user_input = input("> ").strip()
            return ML_Copilot(user_input=user_input)
        
    @step
    async def list_files(self, ctx: Context, ev: ListFilesEvent) -> PreprocessEvent | ML_Copilot | ListFilesEvent | StopEvent :
        files = os.listdir('.')
        print("Current directory files:")
        for f in files:
            print(f"- {f}")
        print("What would you like to do next?")
        user_input = input("> ").strip()
        return ML_Copilot(user_input=user_input)

    @step
    async def data_preprocessing(self, ctx: Context, ev: PreprocessEvent) -> TrainEvent | ML_Copilot | ListFilesEvent | StopEvent:
        dataset_path = ev.dataset_path
        target_column = ev.target_column
        save_path = ev.save_path 
        additional_instructions = ev.additional_instructions

        # Prepare the prompt for the agent
        prompt = f"""
We have a dataset at '{dataset_path}'.
Please write and execute Python code to:
- Load the dataset into a pandas DataFrame.
- Preprocess the data for a binary classification task, including:
  - Handle missing values if any.
  - Encode categorical variables.
  - Scale or normalize numerical features (without changing the target labels).
- Ensure the target variable '{target_column}' remains unchanged.
- Save the preprocessed data to '{save_path}'.
"""
        if additional_instructions:
            prompt += f"\nAdditional instructions from the user: {additional_instructions}"

        # Use the agent to generate and execute code asynchronously
        response = await asyncio.get_event_loop().run_in_executor(
            None, self.agent.chat, prompt
        )

        # Print the agent's response
        print(response)

        # Store the path to the preprocessed data in the context
        await ctx.set('preprocessed_data_path', save_path)

        print("Data preprocessing is complete. What would you like to train the model next ?: type : 'train' ")
        user_input = input("> ").strip()
        return ML_Copilot(user_input=user_input)

    @step
    async def training(self, ctx: Context, ev: TrainEvent) -> EvaluateEvent | ML_Copilot | ListFilesEvent | DocumentEvent | StopEvent:
        # Retrieve the preprocessed data path from the context
        preprocessed_data_path = await ctx.get('preprocessed_data_path', default=None)
        if not preprocessed_data_path:
            print("Preprocessed data not found. Please run preprocessing first.")
            user_input = input("> ").strip()
            return ML_Copilot(user_input=user_input)

        model_path = ev.model_path
        additional_instructions = ev.additional_instructions
        # Prepare the prompt for the agent
        prompt = f"""
We have preprocessed data at '{preprocessed_data_path}'.
Please write and execute Python code to:
- Load the preprocessed data.
- Split the data into training and test sets.
- Train a binary classification model using an appropriate algorithm (e.g., Logistic Regression, SVM, Random Forest).
- Save the trained model to '{model_path}'.
"""
        if additional_instructions:
            prompt += f"\nAdditional instructions from the user: {additional_instructions}"

        # Use the agent to generate and execute code asynchronously
        response = await asyncio.get_event_loop().run_in_executor(
            None, self.agent.chat, prompt
        )

        # Print the agent's response
        print(response)

        # Store the model path in the context
        # await ctx.set('model_path', 'models/model.pkl')
        await ctx.set('model_path', model_path)


        print("Model training is complete. What would you like to evaluate the model next? type: 'evaluate' ")
        user_input = input("> ").strip()
        return ML_Copilot(user_input=user_input)

    @step
    async def evaluation(self, ctx: Context, ev: EvaluateEvent) -> PlotEvent | ML_Copilot | ListFilesEvent | DocumentEvent | StopEvent:
        # Retrieve the model path and preprocessed data path from the context
        model_path = await ctx.get('model_path', default=None)
        preprocessed_data_path = await ctx.get('preprocessed_data_path', default=None)
        evaluation_save_path = ev.evaluation_save_path
        additional_instructions = ev.additional_instructions

        if not model_path or not preprocessed_data_path:
            print("Model or preprocessed data not found. Please ensure both are available.")
            user_input = input("> ").strip()
            return ML_Copilot(user_input=user_input)
        
        # Ask the user if they want to use a different model
        print(f"Current model path is '{model_path}'. Do you want to use a different model for evaluation? (yes/no)")
        user_response = input("> ").strip().lower()
        if user_response in ('yes', 'y'):
            print("Please enter the model path:")
            model_path = input("> ").strip()
            await ctx.set('model_path', model_path)

        # Prepare the prompt for the agent
        prompt = f"""
We have a trained model at '{model_path}' and preprocessed data at '{preprocessed_data_path}'.
Please write and execute Python code to:
- Load the preprocessed data and the trained model.
- Evaluate the model on the test set.
- Provide evaluation metrics including accuracy, precision, recall, and F1-score, AUC.
- Save the evaluation results to '{evaluation_save_path}'.
"""
        if additional_instructions:
            prompt += f"\nAdditional instructions from the user: {additional_instructions}"

        # Use the agent to generate and execute code asynchronously
        response = await asyncio.get_event_loop().run_in_executor(
            None, self.agent.chat, prompt
        )

        # Print the agent's response
        print(response)

        # Store the evaluation results path in the context
        # await ctx.set('evaluation_results_path', 'results/evaluation.txt')
        await ctx.set('evaluation_results_path', evaluation_save_path)

        print("Model evaluation is complete. What would you like to do plot metrics value next? type : 'plot' ")
        user_input = input("> ").strip()
        return ML_Copilot(user_input=user_input)

    @step
    async def plotting(self, ctx: Context, ev: PlotEvent) -> ML_Copilot | ListFilesEvent | DocumentEvent | StopEvent:
        if ev.plot_type == 'results':
            await self.plot_results(ctx, ev)
        elif ev.plot_type == 'data':
            await self.plot_data(ctx, ev)
        else:
            print("Invalid plot type.")
            user_input = input("> ").strip()
            return ML_Copilot(user_input=user_input)
        
        # After plotting, prompt for next action
        print("Plots generated and saved to 'results'. What would you like to do next?")
        user_input = input("> ").strip()
        return ML_Copilot(user_input=user_input)

    # @step 
    async def plot_results(self, ctx: Context, ev: PlotEvent) -> ML_Copilot | ListFilesEvent | DocumentEvent | StopEvent:
        # Retrieve the evaluation results path from the context
        evaluation_results_path = await ctx.get('evaluation_results_path', default=None)
        if evaluation_results_path:
            print(f"Default evaluation results path is '{evaluation_results_path}'. Do you want to use this path? (yes/no)")
            user_response = input("> ").strip().lower()
            if user_response in ('yes', 'y'):
                data_file_path = evaluation_results_path
            else:
                print("Please enter the data file path for evaluation results:")
                data_file_path = input("> ").strip()
        else:
            print("No evaluation results found. Please provide the data file path for evaluation results:")
            data_file_path = input("> ").strip()
        
        print("Any additional plotting instructions? (e.g., 'plot ROC curve and save as roc_curve.png') (press Enter to skip):")
        additional_instructions = input("> ").strip()
        
        # Ensure the data file exists
        if not os.path.exists(data_file_path):
            print(f"Data file '{data_file_path}' not found.")
            user_input = input("> ").strip()
            return ML_Copilot(user_input=user_input)
        
        # Create 'results' directory if it doesn't exist
        os.makedirs('results', exist_ok=True)
        
        # Prepare the prompt for the agent
        prompt = f"""
    We have evaluation results at '{data_file_path}'.
    Please write and execute Python code to:
    - Load the evaluation results.
    - {additional_instructions if additional_instructions else "Create appropriate plots to visualize evaluation metrics such as accuracy, precision, recall, and ROC curve."}
    - Save the plots to the 'results/' directory with descriptive filenames with the type of the plot.
    """
        # Use the agent to generate and execute code asynchronously
        response = await asyncio.get_event_loop().run_in_executor(
            None, self.agent.chat, prompt
        )
        
        # Print the agent's response
        print(response)
        
        # # Indicate completion
        # print("Plots generated and saved to 'results'. What would you like to do next?")
        # user_input = input("> ").strip()
        # return ML_Copilot(user_input=user_input)

    # @step
    async def plot_data(self, ctx: Context, ev: PlotEvent) -> ML_Copilot | ListFilesEvent | DocumentEvent | StopEvent:
        # Retrieve the preprocessed data path from the context
        preprocessed_data_path = await ctx.get('preprocessed_data_path', default=None)
        if preprocessed_data_path:
            print(f"Default preprocessed data path is '{preprocessed_data_path}'. Do you want to use this path? (yes/no)")
            user_response = input("> ").strip().lower()
            if user_response in ('yes', 'y'):
                data_file_path = preprocessed_data_path
            else:
                print("Please enter the data file path for preprocessed data:")
                data_file_path = input("> ").strip()
        else:
            print("No preprocessed data found. Please provide the data file path for preprocessed data:")
            data_file_path = input("> ").strip()
        
        print("Any additional plotting instructions? (e.g., 'plot feature distributions') (press Enter to skip):")
        additional_instructions = input("> ").strip()
        
        # Ensure the data file exists
        if not os.path.exists(data_file_path):
            print(f"Data file '{data_file_path}' not found.")
            user_input = input("> ").strip()
            return ML_Copilot(user_input=user_input)
        
        # Create 'results' directory if it doesn't exist
        os.makedirs('results', exist_ok=True)
        
        # Prepare the prompt for the agent
        prompt = f"""
    We have preprocessed data at '{data_file_path}'.
    Please write and execute Python code to:
    - Load the data.
    - {additional_instructions if additional_instructions else "Create appropriate plots to visualize data distributions and relationships between features."}
    - Save the plots to the 'results/' directory with descriptive filenames depending on the type of the plot.
    """
        # Use the agent to generate and execute code asynchronously
        response = await asyncio.get_event_loop().run_in_executor(
            None, self.agent.chat, prompt
        )
        
        # Print the agent's response
        print(response)
        
        # # Indicate completion
        # print("Plots generated and saved to 'results'. What would you like to do next?")
        # user_input = input("> ").strip()
        # return ML_Copilot(user_input=user_input)


    @step
    async def documentation(self, ctx: Context, ev: DocumentEvent) -> ML_Copilot | ListFilesEvent | StopEvent:
        # Retrieve paths from the context
        model_path = await ctx.get('model_path', default=None)
        evaluation_results_path = await ctx.get('evaluation_results_path', default=None)

        if not model_path or not evaluation_results_path:
            print("Model or evaluation results not found. Please ensure both are available.")
            user_input = input("> ").strip()
            return ML_Copilot(user_input=user_input)

        # Prepare the prompt for the agent
        prompt = f"""
We have a trained model at '{model_path}' and evaluation results at '{evaluation_results_path}'.
Please write a documentation report summarizing:
- The preprocessing steps.
- The model training process.
- The evaluation results.
Save the report to '{report_save_path}'.
"""

        # Use the agent to generate and execute code asynchronously
        response = await asyncio.get_event_loop().run_in_executor(
            None, self.agent.chat, prompt
        )

        # Print the agent's response
        print(response)

        # Indicate completion
        print("Documentation report generated successfully. What would you like to do next?")
        user_input = input("> ").strip()
        return ML_Copilot(user_input=user_input)

    @step
    async def stop_workflow(self, ctx: Context, ev: StopEvent) -> StopEvent:
        print(ev.result)

# Run the Workflow
async def main():
    workflow = MLWorkflow(timeout=600, verbose=True)
    await workflow.run()

if __name__ == "__main__":
    asyncio.run(main())
