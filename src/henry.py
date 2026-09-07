import json
import logging
import os
import sys

import boto3

from botocore.exceptions import ClientError
from datetime import datetime

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
DEFAULT_BEDROCK_MODEL_ID = "meta.llama3-3-70b-instruct-v1:0"
BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", DEFAULT_BEDROCK_MODEL_ID)
MAX_HISTORY_MESSAGES = 6
LOG_DIR = "logs"
logger = logging.getLogger("henry")
SYSTEM_PERSONA = ("Your name is Kirsche. You are an adventure heroine fox-girl."
                   " If you get asked a question, see if you have a tool that would help."
                   " If you don't have a tool, try to answer the question yourself.")


def add_two_numbers(a: int, b: int) -> int:
    """
    Add two numbers

    Args:
        a (int): The first number
        b (int): The second number

    Returns:
        int: The sum of the two numbers
    """
    logger.debug("add_two_numbers called with a=%s, b=%s", a, b)
    return int(a) + int(b)


def subtract_two_numbers(a: int, b: int) -> int:
    """
    Subtract two numbers

    Args:
        a (int): The first number
        b (int): The second number

    Returns:
        int: The result of subtracting b from a
    """
    logger.debug("subtract_two_numbers called with a=%s, b=%s", a, b)
    return int(a) - int(b)


def multiply_two_numbers(a: int, b: int) -> int:
    """
    Multiply two numbers

    Args:
        a (int): The first number
        b (int): The second number

    Returns:
        int: The product of the two numbers
    """
    logger.debug("multiply_two_numbers called with a=%s, b=%s", a, b)
    return int(a) * int(b)


def divide_two_numbers(a: int, b: int) -> float:
    """
    Divide two numbers

    Args:
        a (int): The numerator
        b (int): The denominator

    Returns:
        float: The result of dividing a by b
    """
    logger.debug("divide_two_numbers called with a=%s, b=%s", a, b)
    return int(a) / int(b)


def capitalize_string(s: str) -> str:
    """
    Capitalize a string

    Args:
        s (str): The string to capitalize

    Returns:
        str: The capitalized string
    """
    logger.debug("capitalize_string called with s=%s", s)
    return str(s).upper()


def get_current_time(*args, **kwargs) -> str:
    """
    Get the current time

    Returns:
        str: The current time in ISO format
    """
    logger.debug("get_current_time called")
    return datetime.now().isoformat()


# Registry of tools available to the LLM: name -> (callable, description, JSON input schema).
TOOLS = {
    "add_two_numbers": {
        "function": add_two_numbers,
        "description": "Add two numbers together.",
        "schema": {
            "type": "object",
            "properties": {
                "a": {"type": "integer", "description": "The first number"},
                "b": {"type": "integer", "description": "The second number"},
            },
            "required": ["a", "b"],
        },
    },
    "subtract_two_numbers": {
        "function": subtract_two_numbers,
        "description": "Subtract the second number from the first number.",
        "schema": {
            "type": "object",
            "properties": {
                "a": {"type": "integer", "description": "The first number"},
                "b": {"type": "integer", "description": "The second number"},
            },
            "required": ["a", "b"],
        },
    },
    "multiply_two_numbers": {
        "function": multiply_two_numbers,
        "description": "Multiply two numbers together.",
        "schema": {
            "type": "object",
            "properties": {
                "a": {"type": "integer", "description": "The first number"},
                "b": {"type": "integer", "description": "The second number"},
            },
            "required": ["a", "b"],
        },
    },
    "divide_two_numbers": {
        "function": divide_two_numbers,
        "description": "Divide the first number by the second number.",
        "schema": {
            "type": "object",
            "properties": {
                "a": {"type": "integer", "description": "The numerator"},
                "b": {"type": "integer", "description": "The denominator"},
            },
            "required": ["a", "b"],
        },
    },
    "capitalize_string": {
        "function": capitalize_string,
        "description": "Convert a string to uppercase.",
        "schema": {
            "type": "object",
            "properties": {
                "s": {"type": "string", "description": "The string to capitalize"},
            },
            "required": ["s"],
        },
    },
    "get_current_time": {
        "function": get_current_time,
        "description": "Get the current date and time in ISO format.",
        "schema": {"type": "object", "properties": {}, "required": []},
    },
}


def describe_tools() -> str:
    """
    Build a human-readable bullet list of the available tools.

    Returns:
        str: A newline-separated list of "name: description" entries.
    """
    logger.debug("describe_tools called")
    return "\n".join(f"- {name}: {spec['description']}" for name, spec in TOOLS.items())


def build_tool_config(tool_names) -> dict:
    """
    Build a Bedrock Converse API toolConfig restricted to the given tool names.

    Args:
        tool_names (list[str]): The names of the tools to include.

    Returns:
        dict: A toolConfig payload suitable for the Converse API.
    """
    logger.debug("build_tool_config called with tool_names=%s", tool_names)
    return {
        "tools": [
            {
                "toolSpec": {
                    "name": name,
                    "description": TOOLS[name]["description"],
                    "inputSchema": {"json": TOOLS[name]["schema"]},
                }
            }
            for name in tool_names
        ]
    }


def decide_tool(bedrock_client, model_id: str, user_msg: str) -> str | None:
    """
    Stage 1: ask the LLM whether the user's message needs one of the available tools.

    Args:
        bedrock_client: The Bedrock Runtime client.
        model_id (str): The model identifier to invoke.
        user_msg (str): The user's current message.

    Returns:
        str | None: The chosen tool name, or None if no tool is needed.
    """
    logger.debug("decide_tool called with user_msg=%s", user_msg)
    system_prompt = (
        "You are a routing assistant. Given the list of available tools below and a user's message,"
        " decide whether answering the message requires calling one of the tools.\n"
        f"Available tools:\n{describe_tools()}\n"
        "Respond with only the exact tool name to use, or respond with NONE if no tool is needed."
        " Do not include any other text."
    )
    response = bedrock_client.converse(
        modelId=model_id,
        messages=[{"role": "user", "content": [{"text": user_msg}]}],
        system=[{"text": system_prompt}],
        inferenceConfig={"maxTokens": 20, "temperature": 0},
    )
    answer = response["output"]["message"]["content"][0]["text"].strip()
    return None if answer.upper() == "NONE" or answer not in TOOLS else answer


def formulate_and_run_tool(bedrock_client, model_id: str, user_msg: str, tool_name: str):
    """
    Stage 2: ask the LLM to formulate arguments for the chosen tool, then run it locally.

    Args:
        bedrock_client: The Bedrock Runtime client.
        model_id (str): The model identifier to invoke.
        user_msg (str): The user's current message.
        tool_name (str): The name of the tool chosen in stage 1.

    Returns:
        tuple[dict, object] | None: The assistant's toolUse message and the tool's result,
            or None if the model did not end up calling the tool.
    """
    logger.debug("formulate_and_run_tool called with tool_name=%s", tool_name)
    response = bedrock_client.converse(
        modelId=model_id,
        messages=[{"role": "user", "content": [{"text": user_msg}]}],
        system=[{"text": f"Use the {tool_name} tool to answer the user's message."}],
        inferenceConfig={"maxTokens": 200, "temperature": 0},
        toolConfig=build_tool_config([tool_name]),
    )
    assistant_message = response["output"]["message"]
    tool_use = next(
        (block["toolUse"] for block in assistant_message["content"] if "toolUse" in block),
        None,
    )
    if tool_use is None:
        return None

    tool_result = TOOLS[tool_name]["function"](**tool_use["input"])
    return assistant_message, tool_use, tool_result


def get_final_answer(bedrock_client, model_id: str, messages: list, tool_config: dict | None = None) -> str:
    """
    Ask the LLM for a plain-text answer given a conversation.

    Args:
        bedrock_client: The Bedrock Runtime client.
        model_id (str): The model identifier to invoke.
        messages (list): The conversation to send.
        tool_config (dict | None): Required if `messages` contains toolUse/toolResult blocks.

    Returns:
        str: The assistant's text reply.
    """
    logger.debug("get_final_answer called")
    converse_kwargs = {
        "modelId": model_id,
        "messages": messages,
        "system": [{"text": SYSTEM_PERSONA}],
        "inferenceConfig": {"maxTokens": 1500, "temperature": 0.5},
    }
    if tool_config is not None:
        converse_kwargs["toolConfig"] = tool_config
    response = bedrock_client.converse(**converse_kwargs)
    # The model may reply with a toolUse block instead of text; skip past it if so.
    content_blocks = response["output"]["message"]["content"]
    text_block = next((block["text"] for block in content_blocks if "text" in block), None)
    if text_block is None:
        logger.warning("get_final_answer got no text block: %s", content_blocks)
        return "I'm not sure how to answer that."
    return text_block


def setup_logging() -> logging.Logger:
    """
    Configure a logger that writes timestamped entries to a dated log file.

    Returns:
        logging.Logger: The configured logger.
    """
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, f"{datetime.now().strftime('%Y-%m-%d')}-henry.log")
    logger = logging.getLogger("henry")
    logger.setLevel(logging.DEBUG)
    handler = logging.FileHandler(log_path)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    return logger


def trim_history(conversation_history: list) -> None:
    """
    Keep only the most recent messages, dropping the oldest ones in place.

    Args:
        conversation_history (list): The conversation history to trim.
    """
    logger.debug("trim_history called")
    while len(conversation_history) > MAX_HISTORY_MESSAGES:
        conversation_history.pop(0)


def handle_prompt(bedrock_client, model_id: str, user_msg: str, conversation_history: list) -> str:
    """
    Run the two-stage tool-use flow for a single user message and return the reply text.

    Only the original prompt and the final reply are appended to `conversation_history`;
    any intermediate toolUse/toolResult exchange is kept out of the persisted history.

    Args:
        bedrock_client: The Bedrock Runtime client.
        model_id (str): The model identifier to invoke.
        user_msg (str): The user's current message.
        conversation_history (list): The conversation history, updated in place.

    Returns:
        str: The assistant's reply text.
    """
    logger.debug("handle_prompt called with user_msg=%s", user_msg)
    user_message = {"role": "user", "content": [{"text": user_msg}]}
    turn_messages = conversation_history + [user_message]

    tool_name = decide_tool(bedrock_client, model_id, user_msg)
    if tool_name is not None:
        tool_call = formulate_and_run_tool(bedrock_client, model_id, user_msg, tool_name)
        if tool_call is not None:
            assistant_message, tool_use, tool_result = tool_call
            turn_messages = turn_messages + [
                assistant_message,
                {
                    "role": "user",
                    "content": [{
                        "toolResult": {
                            "toolUseId": tool_use["toolUseId"],
                            "content": [{"text": json.dumps(tool_result)}],
                        }
                    }],
                },
            ]
            reply = get_final_answer(
                bedrock_client, model_id, turn_messages, tool_config=build_tool_config(list(TOOLS.keys()))
            )
            conversation_history.append(user_message)
            conversation_history.append({"role": "assistant", "content": [{"text": reply}]})
            return reply

    reply = get_final_answer(bedrock_client, model_id, turn_messages)
    conversation_history.append(user_message)
    conversation_history.append({"role": "assistant", "content": [{"text": reply}]})
    return reply


def main():
    setup_logging()
    logger.debug("main called")
    bedrock_client = boto3.client("bedrock-runtime", region_name=AWS_REGION)
    # Specify the Llama 3.3 70B Instruct inference profile ID (cross-region)
    model_id = ".".join([AWS_REGION.split("-")[0], BEDROCK_MODEL_ID])

    conversation_history = []

    try:
        while True:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            msg = input(f"[{timestamp}] Your message: ")
            if msg.lower() == "exit":
                break

            logger.info("Prompt: %s", msg)
            reply = handle_prompt(bedrock_client, model_id, msg, conversation_history)
            trim_history(conversation_history)
            logger.info("Output: %s", reply)
            print(f"[{timestamp}] Kirsche: {reply}")
        return 0
    except ClientError as e:
        logger.error("ClientError: %s", e.response['Error']['Message'])
        print(f"Error: {e.response['Error']['Message']}")
        return 1
    except Exception as e:
        logger.error("Unexpected error: %s", e)
        print(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
