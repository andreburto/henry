"""Integration tests for the Henry Bedrock harness."""

import sys
import unittest
from pathlib import Path

import boto3


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import henry


class HandlePromptTest(unittest.TestCase):
    """Verify that prompts are processed through the Bedrock-backed flow."""

    @classmethod
    def setUpClass(cls) -> None:
        """Configure test logs before the Bedrock client is created."""
        henry.setup_logging(str(Path(__file__).resolve().parent))

    def test_solves_chained_arithmetic_prompt(self) -> None:
        """Return the expected value after performing the requested operations."""
        bedrock_client = boto3.client("bedrock-runtime", region_name=henry.AWS_REGION)
        model_id = ".".join([
            henry.AWS_REGION.split("-")[0],
            henry.BEDROCK_MODEL_ID,
        ])
        prompt = (
            "Add 2 and three. Take that sum and multiply it by 5. Subtract one. "
            "Provide a single number as an answer. No other text."
        )

        reply = henry.handle_prompt(bedrock_client, model_id, prompt, [])

        self.assertEqual(reply.strip(), "24")


if __name__ == "__main__":
    unittest.main()