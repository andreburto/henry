# Instructions for GitHub Copilot

## About

This is an experimental harness that will work with an LLM (Large Language Model) to assist with code generation and other tasks.

## .py files

1. Use snake_case for variable and function names.
2. Use PascalCase for class names.
3. Use 4 spaces for indentation.
4. Include docstrings for all functions and classes.
5. Avoid using global variables, except for constants; prefer passing parameters to functions.
6. Use list comprehensions for creating lists when appropriate.
7. Use DRY (Don't Repeat Yourself) principles to avoid code duplication.
8. Shared constants and functions go in the `utils.py` file.
9. Follow PEP 8 style guidelines for Python code.

## .tf files

1. Use snake_case for variable and resource names.
2. Use 2 spaces for indentation.
3. Include comments to explain the purpose of each resource and variable.
4. All resources should be defined in main.tf, and variables should be defined in variables.tf.
5. All files go in the terraform/ directory, except for modules which go in terraform/modules/ directory.

## / files

1. README.md should include an overview of the project, installation instructions, usage examples, and contribution guidelines.
2. .gitignore should include common patterns for ignoring files such as node_modules/, __pycache__/, .env., and .terraform files.
3. .env should include environment variables needed for local development, but should not be committed to version control. Use .env.example to provide a template for required environment variables.