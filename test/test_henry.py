"""Integration tests for the Henry Bedrock harness."""

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import boto3


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import henry


class FileToolTest(unittest.TestCase):
    """Verify filesystem tools use a sanitized configured base path."""

    def setUp(self) -> None:
        """Configure an isolated base directory for every test."""
        self.temporary_directory = TemporaryDirectory()
        self.original_file_path = henry.HENRY_FILE_PATH
        henry.HENRY_FILE_PATH = self.temporary_directory.name

    def tearDown(self) -> None:
        """Restore the configured base directory and remove temporary files."""
        henry.HENRY_FILE_PATH = self.original_file_path
        self.temporary_directory.cleanup()

    def test_sanitize_path_removes_leading_parent_components(self) -> None:
        """Discard parent-directory traversal at the beginning of a path."""
        self.assertEqual(
            henry.sanitize_path("../../output/result.txt"), "output/result.txt"
        )

    def test_file_tools_read_and_write_under_configured_path(self) -> None:
        """Create nested directories, write text, and read it back."""
        written_path = henry.write_text_to_file(
            "saved text", "../../nested/output/result.txt"
        )
        expected_path = Path(self.temporary_directory.name) / "nested/output/result.txt"

        self.assertEqual(written_path, str(expected_path))
        self.assertTrue(expected_path.is_file())
        self.assertEqual(
            henry.read_text_from_file("../../nested/output/result.txt"), "saved text"
        )

    def test_list_files_returns_recursive_relative_paths(self) -> None:
        """List nested files without including directory paths or metadata."""
        henry.write_text_to_file("first", "top.txt")
        henry.write_text_to_file("second", "nested/deeper/bottom.txt")

        self.assertEqual(
            henry.list_files(), ["nested/deeper/bottom.txt", "top.txt"]
        )


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