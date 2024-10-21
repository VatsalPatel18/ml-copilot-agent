# Frequently Asked Questions

## 1. **Do I need an OpenAI API key to use ML Copilot Agent?**

**Yes**, an OpenAI API key is required for the agent to function, as it leverages OpenAI's language models to interpret and execute your instructions.

---

## 2. **Can I use this tool for multiclass classification tasks?**

**Currently**, the agent is primarily designed for binary classification tasks. However, you can attempt multiclass classification by providing custom instructions, but support may be limited.

---

## 3. **How secure is my data?**

The agent sends prompts to OpenAI's servers, which may include snippets of your data. **Avoid using sensitive or personal data**. Ensure that any data you use is anonymized or does not contain confidential information.

---

## 4. **Can I customize the machine learning algorithms used?**

**Yes**, you can customize the algorithms by providing additional instructions during the training step. For example, you can specify using a Random Forest classifier, adjust hyperparameters, or choose different evaluation metrics.

---

## 5. **Is there support for regression tasks?**

**Not at the moment**. The agent is focused on classification tasks. Support for regression tasks may be added in future updates.

---

## 6. **I encountered an error during execution. What should I do?**

Ensure that:

- All file paths provided are correct.
- The data is in the expected format (e.g., CSV with appropriate headers).
- You have a stable internet connection.
- Your OpenAI API key is valid and has sufficient quota.

If the issue persists, please open an issue on the project's GitHub repository with detailed information.

---

## 7. **Can I contribute to the project?**

**Absolutely!** We welcome contributions. Please refer to the [Contributing](contributing.md) section for guidelines on how to contribute.

---

## 8. **Is this tool free to use?**

The tool is free for **non-commercial use** under the CC BY-NC-ND 4.0 license. However, using OpenAI's API may incur costs based on their pricing model. Please check [OpenAI's pricing](https://openai.com/pricing) for more details.

---

## 9. **How do I update the agent to the latest version?**

Pull the latest changes from the repository and reinstall any updated dependencies:

```bash
git pull origin master
pip install -r requirements.txt
```

## 10. ** Who do I contact for support or questions?**
Feel free to open an issue on the project's GitHub repository or reach out via email at vatsal1@hawk-franklin-research.com.

## 11. ** Can I use ML Copilot Agent for commercial purposes?**
No, the CC BY-NC-ND 4.0 license prohibits commercial use of the software. For commercial licensing options, please contact the project maintainers.

## 12. ** Does ML Copilot Agent store or collect any of my data?**
The agent itself does not store or collect your data. However, prompts and data snippets are sent to OpenAI's servers for processing. Please review OpenAI's Privacy Policy for more information.

## 13. **What data formats are supported?**
The agent primarily works with CSV files. Ensure your datasets are in CSV format with appropriate headers for features and target variables.

## 14. **How can I add support for new features or tasks?**
You can extend the agent's functionality by modifying the workflow steps in the ml_copilot_agent/workflow.py file. Refer to the Contributing section for guidelines.

## 15. **Is there a graphical user interface (GUI) available?**
Currently, ML Copilot Agent is a command-line tool. There is no GUI available at this time, but it may be considered for future development.

