from __future__ import annotations

import json
import math
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib import error as urllib_error
from urllib import request as urllib_request

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
API_TEMPLATES_DIR = DATA_DIR / "api_templates"
PROMPT_STORE_DIR = DATA_DIR / "prompt_store"
RESPONSE_OUTPUT_DIR = DATA_DIR / "response_output"
OPENAI_API_KEY_ENV = "OPENAI_API_KEY"
OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"
OPENAI_REQUEST_TIMEOUT_SECONDS = 60
OPENAI_POLL_INTERVAL_SECONDS = 2.0
OPENAI_POLL_MAX_SECONDS = 600

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
    "max_output_tokens": "",
    "reasoning": {
        "effort": "",
    },
    "text": {
        "verbosity": "",
    },
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


def save_response_raw_json(response_data: Mapping[str, Any], preferred_name: str = "") -> Path:
    ensure_api_storage_dirs()
    fallback = datetime.now().strftime("%Y%m%d_%H%M%S") + "_raw"
    base_name = f"{preferred_name}_raw" if str(preferred_name).strip() else ""
    safe_name = sanitize_storage_name(base_name, fallback=fallback)
    path = RESPONSE_OUTPUT_DIR / f"{safe_name}.json"
    path.write_text(json.dumps(response_data, indent=2, ensure_ascii=True), encoding="utf-8")
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

    reasoning_config = template_payload.get("reasoning", {})
    if isinstance(reasoning_config, Mapping):
        reasoning_effort = str(_clean_cell_value(reasoning_config.get("effort"))).strip().lower()
        if reasoning_effort:
            payload["reasoning"] = {"effort": reasoning_effort}

    text_config = template_payload.get("text", {})
    if isinstance(text_config, Mapping):
        verbosity = str(_clean_cell_value(text_config.get("verbosity"))).strip().lower()
        if verbosity:
            payload["text"] = {"verbosity": verbosity}

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


def _load_json_response(request: urllib_request.Request, *, timeout_seconds: int | float) -> dict[str, Any]:
    try:
        with urllib_request.urlopen(request, timeout=timeout_seconds) as response:
            raw_body = response.read().decode("utf-8")
    except urllib_error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
        detail = error_body or exc.reason or "unknown error"
        raise RuntimeError(
            f"OpenAI Responses API request failed with HTTP {exc.code}: {detail}"
        ) from exc
    except urllib_error.URLError as exc:
        detail = getattr(exc, "reason", exc)
        raise RuntimeError(f"OpenAI Responses API request failed: {detail}") from exc

    if not raw_body.strip():
        raise RuntimeError("The OpenAI Responses API returned an empty response body.")

    try:
        response_data = json.loads(raw_body)
    except json.JSONDecodeError as exc:
        raise RuntimeError("The OpenAI Responses API returned invalid JSON.") from exc

    if not isinstance(response_data, dict):
        raise RuntimeError("The OpenAI Responses API returned an unexpected payload shape.")

    return response_data


def create_response(payload: Mapping[str, Any], api_key: str) -> dict[str, Any]:
    request_payload = dict(payload)
    request_payload["background"] = True
    request_body = json.dumps(request_payload).encode("utf-8")
    request = urllib_request.Request(
        OPENAI_RESPONSES_URL,
        data=request_body,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    return _load_json_response(request, timeout_seconds=OPENAI_REQUEST_TIMEOUT_SECONDS)


def retrieve_response(response_id: str, api_key: str) -> dict[str, Any]:
    request = urllib_request.Request(
        f"{OPENAI_RESPONSES_URL}/{response_id}",
        method="GET",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    return _load_json_response(request, timeout_seconds=OPENAI_REQUEST_TIMEOUT_SECONDS)


def _is_retryable_timeout_error(exc: BaseException) -> bool:
    message = str(exc).strip().lower()
    if "timed out" in message:
        return True
    if isinstance(exc, TimeoutError):
        return True
    if isinstance(exc, RuntimeError) and "timed out" in message:
        return True
    return False


def _wait_for_terminal_response(initial_response: Mapping[str, Any], api_key: str) -> dict[str, Any]:
    response_data = dict(initial_response)
    response_id = str(response_data.get("id") or "").strip()
    if not response_id:
        return response_data

    deadline = time.monotonic() + OPENAI_POLL_MAX_SECONDS
    terminal_statuses = {"completed", "failed", "cancelled", "incomplete"}
    last_timeout_error = ""

    while str(response_data.get("status") or "").strip().lower() not in terminal_statuses:
        if time.monotonic() >= deadline:
            detail_suffix = f" Last transport timeout: {last_timeout_error}" if last_timeout_error else ""
            raise RuntimeError(
                f"OpenAI Responses API polling timed out after {OPENAI_POLL_MAX_SECONDS} seconds.{detail_suffix}"
            )
        time.sleep(OPENAI_POLL_INTERVAL_SECONDS)
        try:
            response_data = retrieve_response(response_id, api_key)
            last_timeout_error = ""
        except Exception as exc:
            if _is_retryable_timeout_error(exc) and time.monotonic() < deadline:
                last_timeout_error = str(exc).strip()
                continue
            raise

    return response_data


def _raise_for_terminal_response_issues(response: Mapping[str, Any]) -> None:
    status = str(response.get("status") or "").strip().lower()
    if status == "completed":
        return

    if status == "incomplete":
        incomplete_details = response.get("incomplete_details")
        reason = ""
        if isinstance(incomplete_details, Mapping):
            reason = str(incomplete_details.get("reason") or "").strip()
        if reason:
            raise RuntimeError(f"The Responses API run ended incomplete: {reason}.")
        raise RuntimeError("The Responses API run ended incomplete.")

    error_payload = response.get("error")
    if isinstance(error_payload, Mapping):
        message = str(error_payload.get("message") or "").strip()
        if message:
            raise RuntimeError(f"The Responses API run failed: {message}")

    if status:
        raise RuntimeError(f"The Responses API run ended with status '{status}'.")


def run_responses_request(template_payload: Mapping[str, Any]) -> tuple[dict[str, Any], Any, str]:
    api_key = os.environ.get(OPENAI_API_KEY_ENV, "").strip()
    if not api_key:
        raise RuntimeError(
            f"Set the {OPENAI_API_KEY_ENV} environment variable before triggering a request."
        )

    payload = build_responses_payload(template_payload)
    initial_response = create_response(payload, api_key)
    response = _wait_for_terminal_response(initial_response, api_key)
    default_output_name = str(template_payload.get("default_output_name", "") or "").strip()
    save_response_raw_json(response, default_output_name)
    _raise_for_terminal_response_issues(response)
    output_text = extract_response_output_text(response)
    if not output_text:
        raise RuntimeError("The Responses API call completed but returned no final text output.")
    return payload, response, output_text
