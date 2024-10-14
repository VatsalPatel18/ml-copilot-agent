# ML-Copilot-Agent

ML-Copilot is an interactive machine learning assistant that streamlines the process of data preprocessing, model training, evaluation, plotting results, and generating documentation—all through a command-line interface powered by OpenAI's GPT 4o.

The framework is build as an llm-agent with llama-index workflow, which is able to execute realtime code through code-intepreter which is present as a tool with the llm-agenmt. 

### How to use ? : watch short video (Recommended)
<!-- ![ML-Copilot-Agent Usage]("assets/ml_copilot_use2_720.gif") -->
[![How to use ?: Watch short video](https://img.youtube.com/vi/rci7WLu7Lw8/0.jpg)](https://youtu.be/rci7WLu7Lw8)
<!-- [![Watch the video](https://img.youtube.com/vi/rci7WLu7Lw8)](https://youtube.com/embed/rci7WLu7Lw8) -->

## LLM-Agent-Features

- **List Files**: View files in the current directory.
- **Data Preprocessing**: Preprocess data for binary classification tasks with customizable instructions.
- **Model Training**: Train binary classification models using algorithms like Logistic Regression, SVM, or Random Forest.
- **Model Evaluation**: Evaluate trained models and obtain metrics such as accuracy, precision, recall, F1-score, and AUC.
- **Plotting**: Generate various plots (e.g., bar plots, PCA plots, correlation matrices) from data or evaluation results.
<!-- - **Documentation**: Automatically generate a documentation report summarizing the entire workflow. -->
- **Interactive Workflow**: Seamlessly navigate through different steps with an intuitive command-line interface.
- **Custom Instruction**: Can provide custom instruction to execute code, for example "Perform pca on temp_csv, save the pca results, plot the pca and save plot inside results folder  

Please make a conda environment before begin

## Direct Installation

```bash
pip install ml-copilot-agent
```

### Manual Installation

1. **Clone the repository**:
```bash
git clone https://github.com/VatsalPatel18/ml-copilot-agent.git
```

2. **Navigate to the project directory**:
```bash
cd ml-copilot
```

3. **Install the required dependencies**:
```bash
pip install -r requirements.txt
```
Ensure you have Python 3.7 or higher installed on your system.


## Usage

### Download the file for test run 
```bash
wget https://raw.githubusercontent.com/VatsalPatel18/ml-copilot-agent/master/temp_csv1.data
```

1. **Get Your OpenAI API Key:**:

To use the OpenAI API, you need to obtain your API key. If you haven't done so yet, follow these steps:
- Go to the [OpenAI API keys page](https://platform.openai.com/account/api-keys)
- Log in to your OpenAI account (or sign up if you don't have one).
- Create a new API key and copy it

Remember to delete the key after use. 

2. **Run ML-Copilot**:
```bash
python -m ml_copilot_agent paste-your-openai-api-key
```

Replace `paste-your-openai-api-key` with your actual OpenAI API key.

3. **Interact with ML-Copilot**:

Once started, ML-Copilot will prompt you for commands. You can enter any of the following commands:

- `show files`: Show files in the current directory.
- `preprocess`: Preprocess data for a binary classification task.
- `train`: Train a binary classification model.
- `evaluate`: Evaluate the trained model.
- `plot`: Generate plots from data or evaluation results.
- `document`: Generate a documentation report. (Under Development)
- `exit`: Terminate the workflow.

### Example Workflow

**Step 1: List Files**

```
list files
```

View all files in the current directory to ensure your dataset is available.

**Step 2: Preprocess Data**
```
preprocess
```

- **Dataset Path**: Provide the path to your dataset (e.g., `data/dataset.csv`).
- **Target Column Name**: Specify the name of the target column in your dataset.
- **Save Path**: Choose where to save the preprocessed data (default is `data/preprocessed_data.csv`).
- **Additional Instructions**: (Optional) Add any specific preprocessing instructions (e.g., "use standard scaler").

**Step 3: Train Model**
```
train
```

- **Model Save Path**: Specify where to save the trained model (default is `models/model.pkl`).
- **Additional Instructions**: (Optional) Specify model preferences (e.g., "use SVM classifier").

**Step 4: Evaluate Model**
```
evaluate
```

- **Evaluation Save Path**: Specify where to save the evaluation results (default is `results/evaluation.txt`).

**Step 5: Plot Results**

```
plot
```

- **Data File Path**: Provide the data file path or press Enter to use default evaluation results or preprocessed data.
- **Additional Plotting Instructions**: (Optional) Specify the type of plot (e.g., "make a bar plot of accuracy and precision").

**Step 7: Custom Instructions**

```
custom instruction
```

- Provide any kind of custom instruction that you would like to execute code.

**Step 8: Exit**

```
exit
```

Terminate the workflow when you are done.

## Dependencies

- **Python 3.7 or higher**
- **OpenAI GPT Models**
- **LlamaIndex**
- **Pandas**
- **Scikit-learn**
- **Matplotlib**
- **Seaborn**

Install all dependencies using:
```bash
pip install -r requirements.txt
```
## Project Structure

- `ml_copilot/`
- `__init__.py`: Initialization and configuration.
- `__main__.py`: Entry point of the application.
- `workflow.py`: Defines the MLWorkflow class and all associated steps and events.
- `data/`: Directory where preprocessed data is saved.
- `models/`: Directory where trained models are saved.
- `results/`: Directory where evaluation results and plots are saved.
- `reports/`: Directory where documentation reports are saved.
- `requirements.txt`: Contains all Python dependencies.

## Contributing

Contributions are welcome! Please follow these steps:

1. **Fork the repository**.
2. **Create a new branch** for your feature or bug fix.
3. **Commit your changes** with clear and descriptive messages.
4. **Push to your fork** and submit a pull request.

## License

This project is licensed under the CC-BY-NC-ND-4.0 License.

## Acknowledgments

- Thanks to the developers of OpenAI and LlamaIndex for providing the foundational tools that make this project possible.

## Contact

For any questions or suggestions, feel free to open an issue or contact vatsal1804@gmail.com.


