# Installation

## Prerequisites

- **Python 3.7** or higher (Python 3.12 recommended)
- **OpenAI API Key**: Required for accessing OpenAI's language models.

## Steps

### 1. Install via PIP

```bash
pip install ml-copilot-agent
```

### 2. Set Up Environment Variables
Export your OpenAI API key:
```bash
export OPENAI_API_KEY='your-api-key-here'
```
Alternatively, create a .env file in the project root:
```bash
echo "OPENAI_API_KEY='your-api-key-here'" > .env
```

### 3. Run the Agent
```bash
python -m ml_copilot_agent OPENAI_API_KEY
```