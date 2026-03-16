from __future__ import annotations

import json
import math
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
API_TEMPLATES_DIR = DATA_DIR / "api_templates"
PROMPT_STORE_DIR = DATA_DIR / "prompt_store"
RESPONSE_OUTPUT_DIR = DATA_DIR / "response_output"
OPENAI_API_KEY_ENV = "OPENAI_API_KEY"

DEFAULT_TEMPLATE_PAYLOAD: dict[str, Any] = {
    "name": "",
    "model": "gpt-5",
    "developer_text": "",
    "user_text": "",
    "prompt": {
        "id": "",
        "version": "",
        "variables": [],
    },
    "tool_choice": "auto",
    "temperature": 1.0,
    "top_p": 1.0,
    "max_output_tokens": 1200,
    "store": True,
    "parallel_tool_calls": True,
    "metadata": [],
    "web_search": {
        "enabled": False,
        "allowed_domains": [],
        "user_location": {
            "country": "",
            "city": "",
            "region": "",
            "timezone": "",
        },
        "external_web_access": True,
    },
    "file_search": {
        "enabled": False,
        "vector_store_ids": [],
        "max_num_results": 8,
        "include_results": False,
        "ranking_options": {
            "ranker": "",
            "score_threshold": "",
        },
        "filters": {
            "type": "and",
            "rows": [],
        },
    },
    "default_output_name": "",
}

DEFAULT_PROMPT_PAYLOAD: dict[str, str] = {
    "name": "",
    "developer_text": "",
    "user_text": "",
}


def ensure_api_storage_dirs() -> None:
    for directory in (API_TEMPLATES_DIR, PROMPT_STORE_DIR, RESPONSE_OUTPUT_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def clone_default_template_payload() -> dict[str, Any]:
    return json.loads(json.dumps(DEFAULT_TEMPLATE_PAYLOAD))


def clone_default_prompt_payload() -> dict[str, str]:
    return dict(DEFAULT_PROMPT_PAYLOAD)


def sanitize_storage_name(name: str, *, fallback: str) -> str:
    candidate = (name or "").strip()
    if not candidate:
        candidate = fallback
    candidate = re.sub(r"[<>:\"/\\|?*]+", "_", candidate)
    candidate = candidate.strip(" .")
    return candidate or fallback


def list_saved_names(directory: Path, suffix: str) -> list[str]:
    ensure_api_storage_dirs()
    names = [path.stem for path in directory.glob(f"*{suffix}") if path.is_file()]
    return sorted(names, key=str.lower)


def save_json_document(directory: Path, name: str, payload: Mapping[str, Any]) -> Path:
    ensure_api_storage_dirs()
    safe_name = sanitize_storage_name(name, fallback=datetime.now().strftime("%Y%m%d_%H%M%S"))
    path = directory / f"{safe_name}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    return path


def load_json_document(directory: Path, name: str) -> dict[str, Any]:
    path = directory / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def save_output_text(text: str, preferred_name: str = "") -> Path:
    ensure_api_storage_dirs()
    fallback = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = sanitize_storage_name(preferred_name, fallback=fallback)
    path = RESPONSE_OUTPUT_DIR / f"{safe_name}.txt"
    path.write_text(text, encoding="utf-8")
    return path


def load_output_text(name: str) -> str:
    path = RESPONSE_OUTPUT_DIR / f"{name}.txt"
    return path.read_text(encoding="utf-8")


def rows_to_string_map(rows: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    output: dict[str, str] = {}
    for row in rows:
        key = str(_clean_cell_value(row.get("key"))).strip()
        value = str(_clean_cell_value(row.get("value"))).strip()
        if key and value:
            output[key] = value
    return output


def _clean_cell_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return value


def parse_string_list(values: Any) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        parts = re.split(r"[\n,]+", values)
        return [part.strip() for part in parts if part.strip()]
    if isinstance(values, Iterable):
        output: list[str] = []
        for item in values:
            cleaned = str(_clean_cell_value(item)).strip()
            if cleaned:
                output.append(cleaned)
        return output
    return []


def _parse_filter_value(raw_value: Any, value_type: str, operator: str) -> Any:
    text = str(_clean_cell_value(raw_value)).strip()
    normalized_type = (value_type or "string").strip().lower()
    normalized_operator = (operator or "eq").strip().lower()

    if normalized_type == "number":
        number_value = float(text)
        return int(number_value) if number_value.is_integer() else number_value

    if normalized_type == "boolean":
        lowered = text.lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
        raise ValueError(f"Boolean filter values must be true/false, received '{text}'.")

    if normalized_type == "array" or normalized_operator in {"in", "nin"}:
        if text.startswith("["):
            parsed = json.loads(text)
            if not isinstance(parsed, list):
                raise ValueError("Array filter values must decode to a JSON list.")
            return parsed
        return parse_string_list(text)

    return text


def build_file_search_filters(filter_config: Mapping[str, Any]) -> dict[str, Any] | None:
    rows = filter_config.get("rows", [])
    if not isinstance(rows, Sequence):
        return None

    comparisons: list[dict[str, Any]] = []
    for row in rows:
        key = str(_clean_cell_value(row.get("key"))).strip()
        operator = str(_clean_cell_value(row.get("type") or row.get("operator") or "eq")).strip().lower()
        value_type = str(_clean_cell_value(row.get("value_type") or "string")).strip().lower()
        raw_value = _clean_cell_value(row.get("value"))
        if not key or raw_value == "":
            continue
        comparisons.append(
            {
                "type": operator,
                "key": key,
                "value": _parse_filter_value(raw_value, value_type, operator),
            }
        )

    if not comparisons:
        return None
    if len(comparisons) == 1:
        return comparisons[0]

    compound_type = str(_clean_cell_value(filter_config.get("type") or "and")).strip().lower()
    if compound_type not in {"and", "or"}:
        compound_type = "and"
    return {"type": compound_type, "filters": comparisons}


def build_web_search_tool(config: Mapping[str, Any]) -> dict[str, Any] | None:
    if not config.get("enabled"):
        return None

    tool: dict[str, Any] = {"type": "web_search"}
    external_web_access = config.get("external_web_access")
    if external_web_access is not None:
        tool["external_web_access"] = bool(external_web_access)

    allowed_domains = parse_string_list(config.get("allowed_domains"))
    if allowed_domains:
        tool["filters"] = {"allowed_domains": allowed_domains}

    user_location_source = config.get("user_location", {})
    if isinstance(user_location_source, Mapping):
        location_fields = {
            "country": str(_clean_cell_value(user_location_source.get("country"))).strip(),
            "city": str(_clean_cell_value(user_location_source.get("city"))).strip(),
            "region": str(_clean_cell_value(user_location_source.get("region"))).strip(),
            "timezone": str(_clean_cell_value(user_location_source.get("timezone"))).strip(),
        }
        location_fields = {key: value for key, value in location_fields.items() if value}
        if location_fields:
            location_fields["type"] = "approximate"
            tool["user_location"] = location_fields

    return tool


def build_file_search_tool(config: Mapping[str, Any]) -> tuple[dict[str, Any] | None, list[str]]:
    if not config.get("enabled"):
        return None, []

    vector_store_ids = parse_string_list(config.get("vector_store_ids"))
    if not vector_store_ids:
        raise ValueError("File search requires at least one vector store ID.")

    tool: dict[str, Any] = {
        "type": "file_search",
        "vector_store_ids": vector_store_ids,
    }

    max_num_results = config.get("max_num_results")
    if max_num_results not in (None, ""):
        tool["max_num_results"] = int(max_num_results)

    filters = build_file_search_filters(config.get("filters", {}))
    if filters:
        tool["filters"] = filters

    ranking_config = config.get("ranking_options", {})
    if isinstance(ranking_config, Mapping):
        ranking_options: dict[str, Any] = {}
        ranker = str(_clean_cell_value(ranking_config.get("ranker"))).strip()
        score_threshold = _clean_cell_value(ranking_config.get("score_threshold"))
        if ranker:
            ranking_options["ranker"] = ranker
        if score_threshold not in ("", None):
            ranking_options["score_threshold"] = float(score_threshold)
        if ranking_options:
            tool["ranking_options"] = ranking_options

    include: list[str] = []
    if config.get("include_results"):
        include.append("file_search_call.results")

    return tool, include


def build_prompt_reference(prompt_config: Mapping[str, Any]) -> dict[str, Any] | None:
    prompt_id = str(_clean_cell_value(prompt_config.get("id"))).strip()
    if not prompt_id:
        return None

    prompt: dict[str, Any] = {"id": prompt_id}
    version = str(_clean_cell_value(prompt_config.get("version"))).strip()
    if version:
        prompt["version"] = version

    variables_rows = prompt_config.get("variables", [])
    if isinstance(variables_rows, Sequence):
        variables = rows_to_string_map(variables_rows)
        if variables:
            prompt["variables"] = variables

    return prompt


def build_responses_payload(template_payload: Mapping[str, Any]) -> dict[str, Any]:
    model = str(_clean_cell_value(template_payload.get("model"))).strip()
    if not model:
        raise ValueError("Model is required.")

    payload: dict[str, Any] = {"model": model}

    prompt_reference = build_prompt_reference(template_payload.get("prompt", {}))
    if prompt_reference:
        payload["prompt"] = prompt_reference

    developer_text = str(_clean_cell_value(template_payload.get("developer_text"))).strip()
    if developer_text:
        payload["instructions"] = developer_text

    user_text = str(_clean_cell_value(template_payload.get("user_text"))).strip()
    if user_text:
        payload["input"] = user_text

    metadata_rows = template_payload.get("metadata", [])
    if isinstance(metadata_rows, Sequence):
        metadata = rows_to_string_map(metadata_rows)
        if metadata:
            payload["metadata"] = metadata

    for numeric_field in ("temperature", "top_p"):
        value = _clean_cell_value(template_payload.get(numeric_field))
        if value not in ("", None):
            payload[numeric_field] = float(value)

    max_output_tokens = _clean_cell_value(template_payload.get("max_output_tokens"))
    if max_output_tokens not in ("", None):
        payload["max_output_tokens"] = int(max_output_tokens)

    tool_choice = str(_clean_cell_value(template_payload.get("tool_choice"))).strip()
    if tool_choice:
        payload["tool_choice"] = tool_choice

    for bool_field in ("store", "parallel_tool_calls"):
        value = template_payload.get(bool_field)
        if value is not None:
            payload[bool_field] = bool(value)

    tools: list[dict[str, Any]] = []
    include: list[str] = []

    web_search_tool = build_web_search_tool(template_payload.get("web_search", {}))
    if web_search_tool:
        tools.append(web_search_tool)

    file_search_tool, file_search_include = build_file_search_tool(template_payload.get("file_search", {}))
    if file_search_tool:
        tools.append(file_search_tool)
        include.extend(file_search_include)

    if tools:
        payload["tools"] = tools
    if include:
        payload["include"] = sorted(set(include))

    if "prompt" not in payload and "instructions" not in payload and "input" not in payload:
        raise ValueError("Provide at least a prompt reference, developer instructions, or user input.")

    return payload


def extract_response_output_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    if hasattr(response, "model_dump"):
        data = response.model_dump()
    elif isinstance(response, Mapping):
        data = dict(response)
    else:
        data = response.__dict__

    if isinstance(data.get("output_text"), str) and data["output_text"].strip():
        return data["output_text"].strip()

    output_items = data.get("output", [])
    for item in output_items:
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            text_value = content.get("text")
            if isinstance(text_value, str) and text_value.strip():
                return text_value.strip()
    return ""


def run_responses_request(template_payload: Mapping[str, Any]) -> tuple[dict[str, Any], Any, str]:
    api_key = os.environ.get(OPENAI_API_KEY_ENV, "").strip()
    if not api_key:
        raise RuntimeError(
            f"Set the {OPENAI_API_KEY_ENV} environment variable before triggering a request."
        )

    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise RuntimeError(
            "The openai package is not installed. Run `pip install -r requirements.txt` first."
        ) from exc

    payload = build_responses_payload(template_payload)
    client = OpenAI(api_key=api_key)
    response = client.responses.create(**payload)
    output_text = extract_response_output_text(response)
    if not output_text:
        raise RuntimeError("The Responses API call completed but returned no final text output.")
    return payload, response, output_text
