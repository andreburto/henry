# Henry Harness Project

This project is an experimental harness designed to work with a Large Language Model (LLM) to assist with code generation and other tasks. It includes a set of Python utilities and tools that can be invoked programmatically or through the LLM interface.

This project targets Llama 3.x as the LLM to save me money. :-D

## Usage

1. Install the required dependencies:

```bash
python3 -m pip install -r requirements.txt
```

2. Run the Python script to start using the harness:

```bash
python3 src/henry.py
```

3. Type `exit` to quit the harness when you are done.

## To Do

- Add a feature to allow routing LLM calls to hosts other than AWS.
- Improve tool calling mechanism and error handling.
- Add prompts and logic to guide the LLM to break down complex tasks into manageable steps.

## Update Log

**2026-09-07:** Started the project using an existing POC script. This version has a loop, some tools, and uses AWS Bedrock.