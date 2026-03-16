import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import openai_responses_service as service


class OpenAIResponsesServiceTests(unittest.TestCase):
    def test_build_responses_payload_includes_prompt_tools_and_overrides(self) -> None:
        payload = service.build_responses_payload(
            {
                "name": "earnings_cockpit",
                "model": "gpt-5",
                "developer_text": "Follow the dashboard prompt, but keep it concise.",
                "user_text": "Summarize the latest filing.",
                "prompt": {
                    "id": "pmpt_123",
                    "version": "7",
                    "variables": [{"key": "ticker", "value": "NVDA"}],
                },
                "tool_choice": "auto",
                "temperature": 0.3,
                "top_p": 0.9,
                "max_output_tokens": 700,
                "store": True,
                "parallel_tool_calls": True,
                "metadata": [{"key": "workspace", "value": "equipilot"}],
                "web_search": {
                    "enabled": True,
                    "allowed_domains": "sec.gov\ninvestor.nvidia.com",
                    "external_web_access": False,
                    "user_location": {
                        "country": "RO",
                        "city": "Bucharest",
                        "region": "B",
                        "timezone": "Europe/Bucharest",
                    },
                },
                "file_search": {
                    "enabled": True,
                    "vector_store_ids": "vs_123\nvs_456",
                    "max_num_results": 6,
                    "include_results": True,
                    "ranking_options": {
                        "ranker": "auto",
                        "score_threshold": "0.25",
                    },
                    "filters": {
                        "type": "and",
                        "rows": [
                            {"key": "ticker", "type": "eq", "value_type": "string", "value": "NVDA"},
                            {"key": "year", "type": "gte", "value_type": "number", "value": "2024"},
                        ],
                    },
                },
            }
        )

        self.assertEqual(payload["prompt"]["id"], "pmpt_123")
        self.assertNotIn("prompt_id", payload["prompt"])
        self.assertEqual(payload["instructions"], "Follow the dashboard prompt, but keep it concise.")
        self.assertEqual(payload["input"], "Summarize the latest filing.")
        self.assertEqual(payload["metadata"], {"workspace": "equipilot"})
        self.assertEqual(payload["include"], ["file_search_call.results"])

        tools = {tool["type"]: tool for tool in payload["tools"]}
        self.assertEqual(tools["web_search"]["filters"]["allowed_domains"], ["sec.gov", "investor.nvidia.com"])
        self.assertEqual(tools["web_search"]["user_location"]["timezone"], "Europe/Bucharest")
        self.assertEqual(tools["file_search"]["vector_store_ids"], ["vs_123", "vs_456"])
        self.assertEqual(tools["file_search"]["ranking_options"]["score_threshold"], 0.25)
        self.assertEqual(tools["file_search"]["filters"]["type"], "and")
        self.assertEqual(len(tools["file_search"]["filters"]["filters"]), 2)

    def test_build_file_search_filters_supports_array_and_boolean_types(self) -> None:
        filters = service.build_file_search_filters(
            {
                "type": "or",
                "rows": [
                    {"key": "is_public", "type": "eq", "value_type": "boolean", "value": "true"},
                    {"key": "sector", "type": "in", "value_type": "array", "value": "AI,Semis"},
                ],
            }
        )

        self.assertEqual(filters["type"], "or")
        self.assertEqual(filters["filters"][0]["value"], True)
        self.assertEqual(filters["filters"][1]["value"], ["AI", "Semis"])

    def test_template_and_prompt_documents_round_trip(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            directory = Path(tmp_dir)
            saved_path = service.save_json_document(directory, "my template", {"name": "my template"})
            loaded = service.load_json_document(directory, "my template")
            names = service.list_saved_names(directory, ".json")

        self.assertEqual(saved_path.name, "my template.json")
        self.assertEqual(loaded["name"], "my template")
        self.assertEqual(names, ["my template"])

    def test_save_output_text_uses_custom_name_or_datetime_fallback(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            with patch.object(service, "RESPONSE_OUTPUT_DIR", Path(tmp_dir)):
                custom_path = service.save_output_text("hello world", "manual_name")
                fallback_path = service.save_output_text("second output", "")

                self.assertEqual(custom_path.name, "manual_name.txt")
                self.assertTrue(fallback_path.name.endswith(".txt"))
                self.assertRegex(fallback_path.stem, r"^\d{8}_\d{6}$")
                self.assertEqual(custom_path.read_text(encoding="utf-8"), "hello world")

    def test_extract_response_output_text_prefers_output_text_and_falls_back_to_message_content(self) -> None:
        class FakeResponse:
            output_text = ""

            def model_dump(self) -> dict[str, object]:
                return {
                    "output": [
                        {
                            "type": "message",
                            "content": [
                                {"type": "output_text", "text": "  Final answer from content  "},
                            ],
                        }
                    ]
                }

        response = FakeResponse()
        self.assertEqual(service.extract_response_output_text(response), "Final answer from content")

    def test_run_responses_request_requires_api_key(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, service.OPENAI_API_KEY_ENV):
                service.run_responses_request(
                    {
                        "model": "gpt-5",
                        "developer_text": "You are helpful.",
                    }
                )


if __name__ == "__main__":
    unittest.main()
