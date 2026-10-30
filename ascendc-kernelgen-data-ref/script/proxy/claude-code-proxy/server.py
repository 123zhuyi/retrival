from fastapi import FastAPI, Request, HTTPException
import uvicorn
import logging
import json
from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Any, Optional, Union, Literal
import httpx
import os
import asyncio
import io
import tempfile
import zipfile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
import litellm
import uuid
import time
from dotenv import load_dotenv
import re
from datetime import datetime
import sys
from pathlib import Path

from trace_db import (
    DB_PATH,
    archive_session,
    build_agent_tree,
    build_history_chains,
    build_prefix_trie,
    build_time_trajectory,
    build_timeline,
    clear_traces,
    get_agent_call,
    get_request,
    get_session,
    get_tool_event,
    get_trace_enabled,
    headers_to_trace,
    list_requests,
    list_sessions,
    model_to_plain,
    monotonic_ms,
    new_trace_id,
    record_request_completed as trace_request_completed,
    record_request_failed as trace_request_failed,
    record_request_started as trace_request_started,
    set_trace_enabled,
    snapshot_stats,
    unarchive_session,
)

SCRIPTS_DIR = Path(__file__).parent / "scripts"
EXPORT_TRAINING_SCRIPT = SCRIPTS_DIR / "export_training.py"

import litellm
litellm.ssl_verify = False
# Suppress LiteLLM's red "Provider List: https://docs.litellm.ai/docs/providers"
# print spam. It's emitted from get_llm_provider_logic.py whenever an internal
# provider-resolution attempt fails — often as harmless retries that don't
# affect the main flow. Real BadRequestErrors still propagate normally.
litellm.suppress_debug_info = True
# litellm.drop_params = True

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.WARN,  # Change to INFO level to show more details
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Configure uvicorn to be quieter
import uvicorn
# Tell uvicorn's loggers to be quiet
logging.getLogger("uvicorn").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("uvicorn.error").setLevel(logging.WARNING)

# Create a filter to block any log messages containing specific strings
class MessageFilter(logging.Filter):
    def filter(self, record):
        # Block messages containing these strings
        blocked_phrases = [
            "LiteLLM completion()",
            "HTTP Request:", 
            "selected model name for cost calculation",
            "utils.py",
            "cost_calculator"
        ]
        
        if hasattr(record, 'msg') and isinstance(record.msg, str):
            for phrase in blocked_phrases:
                if phrase in record.msg:
                    return False
        return True

# Apply the filter to the root logger to catch all messages
root_logger = logging.getLogger()
root_logger.addFilter(MessageFilter())

# Custom formatter for model mapping logs
class ColorizedFormatter(logging.Formatter):
    """Custom formatter to highlight model mappings"""
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    RESET = "\033[0m"
    BOLD = "\033[1m"
    
    def format(self, record):
        if record.levelno == logging.debug and "MODEL MAPPING" in record.msg:
            # Apply colors and formatting to model mapping logs
            return f"{self.BOLD}{self.GREEN}{record.msg}{self.RESET}"
        return super().format(record)

# Apply custom formatter to console handler
for handler in logger.handlers:
    if isinstance(handler, logging.StreamHandler):
        handler.setFormatter(ColorizedFormatter('%(asctime)s - %(levelname)s - %(message)s'))

app = FastAPI()

# Get API keys from environment
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# Get Vertex AI project and location from environment (if set)
VERTEX_PROJECT = os.environ.get("VERTEX_PROJECT", "unset")
VERTEX_LOCATION = os.environ.get("VERTEX_LOCATION", "unset")

# Option to use Gemini API key instead of ADC for Vertex AI
USE_VERTEX_AUTH = os.environ.get("USE_VERTEX_AUTH", "False").lower() == "true"

# Get OpenAI base URL from environment (if set)
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL")

# Get Anthropic base URL from environment (used by the passthrough path).
# Defaults to the official Anthropic API. Set this to a compatible endpoint
# (e.g. https://api.deepseek.com/anthropic) to forward requests there.
ANTHROPIC_BASE_URL = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com").rstrip("/")

# Get preferred provider (default to openai)
PREFERRED_PROVIDER = os.environ.get("PREFERRED_PROVIDER", "openai").lower()

# Get model mapping configuration from environment
# Default to latest OpenAI models if not set
BIG_MODEL = os.environ.get("BIG_MODEL", "gpt-4.1")
SMALL_MODEL = os.environ.get("SMALL_MODEL", "gpt-4.1-mini")

# Get default model context limits from environment (optional overrides)
DEFAULT_MAX_INPUT_TOKENS = os.environ.get("DEFAULT_MAX_INPUT_TOKENS")
DEFAULT_MAX_OUTPUT_TOKENS = os.environ.get("DEFAULT_MAX_OUTPUT_TOKENS")
if DEFAULT_MAX_INPUT_TOKENS:
    DEFAULT_MAX_INPUT_TOKENS = int(DEFAULT_MAX_INPUT_TOKENS)
if DEFAULT_MAX_OUTPUT_TOKENS:
    DEFAULT_MAX_OUTPUT_TOKENS = int(DEFAULT_MAX_OUTPUT_TOKENS)

# Test-only mode: record real client requests but skip upstream LLM calls.
TRACE_ECHO_ONLY = os.environ.get("CC_TRACE_ECHO_ONLY", "false").lower() in {"1", "true", "yes", "on"}

# List of OpenAI models
OPENAI_MODELS = [
    "o3-mini",
    "o1",
    "o1-mini",
    "o1-pro",
    "gpt-4.5-preview",
    "gpt-4o",
    "gpt-4o-audio-preview",
    "chatgpt-4o-latest",
    "gpt-4o-mini",
    "gpt-4o-mini-audio-preview",
    "gpt-4.1",  # Added default big model
    "gpt-4.1-mini" # Added default small model
]

# List of Gemini models
GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-pro"
]

# Helper function to clean schema for Gemini
def clean_gemini_schema(schema: Any) -> Any:
    """Recursively removes unsupported fields from a JSON schema for Gemini."""
    if isinstance(schema, dict):
        # Remove specific keys unsupported by Gemini tool parameters
        schema.pop("additionalProperties", None)
        schema.pop("default", None)

        # Check for unsupported 'format' in string types
        if schema.get("type") == "string" and "format" in schema:
            allowed_formats = {"enum", "date-time"}
            if schema["format"] not in allowed_formats:
                logger.debug(f"Removing unsupported format '{schema['format']}' for string type in Gemini schema.")
                schema.pop("format")

        # Recursively clean nested schemas (properties, items, etc.)
        for key, value in list(schema.items()): # Use list() to allow modification during iteration
            schema[key] = clean_gemini_schema(value)
    elif isinstance(schema, list):
        # Recursively clean items in a list
        return [clean_gemini_schema(item) for item in schema]
    return schema

# Models for Anthropic API requests
class ContentBlockText(BaseModel):
    type: Literal["text"]
    text: str

class ContentBlockImage(BaseModel):
    type: Literal["image"]
    source: Dict[str, Any]

class ContentBlockToolUse(BaseModel):
    type: Literal["tool_use"]
    id: str
    name: str
    input: Dict[str, Any]

class ContentBlockToolResult(BaseModel):
    type: Literal["tool_result"]
    tool_use_id: str
    content: Union[str, List[Dict[str, Any]], Dict[str, Any], List[Any], Any]

class ContentBlockThinking(BaseModel):
    type: Literal["thinking"]
    thinking: str
    signature: Optional[str] = None  # DeepSeek includes this

class SystemContent(BaseModel):
    type: Literal["text"]
    text: str

class Message(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: Union[str, List[Union[ContentBlockText, ContentBlockImage, ContentBlockToolUse, ContentBlockToolResult, ContentBlockThinking]]]

class Tool(BaseModel):
    name: str
    description: Optional[str] = None
    input_schema: Dict[str, Any]

class ThinkingConfig(BaseModel):
    type: Literal["enabled", "disabled", "adaptive", "auto"] = "auto"

class MessagesRequest(BaseModel):
    model: str
    max_tokens: int
    messages: List[Message]
    system: Optional[Union[str, List[SystemContent]]] = None
    stop_sequences: Optional[List[str]] = None
    stream: Optional[bool] = False
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[Dict[str, Any]] = None
    thinking: Optional[ThinkingConfig] = None
    original_model: Optional[str] = None  # Will store the original model name
    
    @field_validator('model')
    def validate_model_field(cls, v, info): # Renamed to avoid conflict
        original_model = v
        new_model = v # Default to original value

        logger.debug(f"📋 MODEL VALIDATION: Original='{original_model}', Preferred='{PREFERRED_PROVIDER}', BIG='{BIG_MODEL}', SMALL='{SMALL_MODEL}'")

        # Remove provider prefixes for easier matching
        clean_v = v
        if clean_v.startswith('anthropic/'):
            clean_v = clean_v[10:]
        elif clean_v.startswith('openai/'):
            clean_v = clean_v[7:]
        elif clean_v.startswith('gemini/'):
            clean_v = clean_v[7:]

        # --- Mapping Logic --- START ---
        mapped = False
        if PREFERRED_PROVIDER == "anthropic":
            if 'haiku' in clean_v.lower():
                new_model = f"anthropic/{SMALL_MODEL}"
            elif 'sonnet' in clean_v.lower() or 'opus' in clean_v.lower() or clean_v.startswith('claude-'):
                new_model = f"anthropic/{BIG_MODEL}"
            else:
                new_model = f"anthropic/{clean_v}"
            mapped = True

        # Map Haiku to SMALL_MODEL based on provider preference
        elif 'haiku' in clean_v.lower():
            if PREFERRED_PROVIDER == "google" and SMALL_MODEL in GEMINI_MODELS:
                new_model = f"gemini/{SMALL_MODEL}"
                mapped = True
            else:
                new_model = f"openai/{SMALL_MODEL}"
                mapped = True

        # Map Sonnet to BIG_MODEL based on provider preference
        elif 'sonnet' in clean_v.lower():
            if PREFERRED_PROVIDER == "google" and BIG_MODEL in GEMINI_MODELS:
                new_model = f"gemini/{BIG_MODEL}"
                mapped = True
            else:
                new_model = f"openai/{BIG_MODEL}"
                mapped = True

        # Add prefixes to non-mapped models if they match known lists
        elif not mapped:
            if clean_v in GEMINI_MODELS and not v.startswith('gemini/'):
                new_model = f"gemini/{clean_v}"
                mapped = True # Technically mapped to add prefix
            elif clean_v in OPENAI_MODELS and not v.startswith('openai/'):
                new_model = f"openai/{clean_v}"
                mapped = True # Technically mapped to add prefix
            elif clean_v.startswith('claude-') and not v.startswith('anthropic/'):
                # Other Claude models (like opus) map to BIG_MODEL like sonnet
                if PREFERRED_PROVIDER == "google" and BIG_MODEL in GEMINI_MODELS:
                    new_model = f"gemini/{BIG_MODEL}"
                else:
                    new_model = f"openai/{BIG_MODEL}"
                mapped = True
        # --- Mapping Logic --- END ---

        if mapped:
            logger.debug(f"📌 MODEL MAPPING: '{original_model}' ➡️ '{new_model}'")
        else:
             # If no mapping occurred and no prefix exists, log warning or decide default
             if not v.startswith(('openai/', 'gemini/', 'anthropic/')):
                 logger.warning(f"⚠️ No prefix or mapping rule for model: '{original_model}'. Using as is.")
             new_model = v # Ensure we return the original if no rule applied

        # Store the original model in the values dictionary
        values = info.data
        if isinstance(values, dict):
            values['original_model'] = original_model

        return new_model

class TokenCountRequest(BaseModel):
    model: str
    messages: List[Message]
    system: Optional[Union[str, List[SystemContent]]] = None
    tools: Optional[List[Tool]] = None
    thinking: Optional[ThinkingConfig] = None
    tool_choice: Optional[Dict[str, Any]] = None
    original_model: Optional[str] = None  # Will store the original model name

    @field_validator('model')
    def validate_model_token_count(cls, v, info): # Renamed to avoid conflict
        # Use the same logic as MessagesRequest validator
        # NOTE: Pydantic validators might not share state easily if not class methods
        # Re-implementing the logic here for clarity, could be refactored
        original_model = v
        new_model = v # Default to original value

        logger.debug(f"📋 TOKEN COUNT VALIDATION: Original='{original_model}', Preferred='{PREFERRED_PROVIDER}', BIG='{BIG_MODEL}', SMALL='{SMALL_MODEL}'")

        # Remove provider prefixes for easier matching
        clean_v = v
        if clean_v.startswith('anthropic/'):
            clean_v = clean_v[10:]
        elif clean_v.startswith('openai/'):
            clean_v = clean_v[7:]
        elif clean_v.startswith('gemini/'):
            clean_v = clean_v[7:]

        # --- Mapping Logic --- START ---
        mapped = False
        if PREFERRED_PROVIDER == "anthropic":
            if 'haiku' in clean_v.lower():
                new_model = f"anthropic/{SMALL_MODEL}"
            elif 'sonnet' in clean_v.lower() or 'opus' in clean_v.lower() or clean_v.startswith('claude-'):
                new_model = f"anthropic/{BIG_MODEL}"
            else:
                new_model = f"anthropic/{clean_v}"
            mapped = True

        # Map Haiku to SMALL_MODEL based on provider preference
        elif 'haiku' in clean_v.lower():
            if PREFERRED_PROVIDER == "google" and SMALL_MODEL in GEMINI_MODELS:
                new_model = f"gemini/{SMALL_MODEL}"
                mapped = True
            else:
                new_model = f"openai/{SMALL_MODEL}"
                mapped = True

        # Map Sonnet to BIG_MODEL based on provider preference
        elif 'sonnet' in clean_v.lower():
            if PREFERRED_PROVIDER == "google" and BIG_MODEL in GEMINI_MODELS:
                new_model = f"gemini/{BIG_MODEL}"
                mapped = True
            else:
                new_model = f"openai/{BIG_MODEL}"
                mapped = True

        # Add prefixes to non-mapped models if they match known lists
        elif not mapped:
            if clean_v in GEMINI_MODELS and not v.startswith('gemini/'):
                new_model = f"gemini/{clean_v}"
                mapped = True # Technically mapped to add prefix
            elif clean_v in OPENAI_MODELS and not v.startswith('openai/'):
                new_model = f"openai/{clean_v}"
                mapped = True # Technically mapped to add prefix
            elif clean_v.startswith('claude-') and not v.startswith('anthropic/'):
                # Other Claude models (like opus) map to BIG_MODEL like sonnet
                if PREFERRED_PROVIDER == "google" and BIG_MODEL in GEMINI_MODELS:
                    new_model = f"gemini/{BIG_MODEL}"
                else:
                    new_model = f"openai/{BIG_MODEL}"
                mapped = True
        # --- Mapping Logic --- END ---

        if mapped:
            logger.debug(f"📌 TOKEN COUNT MAPPING: '{original_model}' ➡️ '{new_model}'")
        else:
             if not v.startswith(('openai/', 'gemini/', 'anthropic/')):
                 logger.warning(f"⚠️ No prefix or mapping rule for token count model: '{original_model}'. Using as is.")
             new_model = v # Ensure we return the original if no rule applied

        # Store the original model in the values dictionary
        values = info.data
        if isinstance(values, dict):
            values['original_model'] = original_model

        return new_model

class TokenCountResponse(BaseModel):
    input_tokens: int

class Usage(BaseModel):
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

class ModelInfo(BaseModel):
    """Model capability information to inform Claude Code about actual model limits."""
    max_input_tokens: Optional[int] = None
    max_output_tokens: Optional[int] = None

class MessagesResponse(BaseModel):
    id: str
    model: str
    role: Literal["assistant"] = "assistant"
    content: List[Union[ContentBlockText, ContentBlockToolUse]]
    type: Literal["message"] = "message"
    stop_reason: Optional[Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"]] = None
    stop_sequence: Optional[str] = None
    usage: Usage
    model_info: Optional[ModelInfo] = None  # Add model capability information

def estimate_trace_echo_tokens(value: Any) -> int:
    try:
        text = json.dumps(model_to_plain(value), ensure_ascii=False)
    except Exception:
        text = str(value)
    return max(1, len(text) // 4)


def get_model_context_info(model_name: str) -> Optional[ModelInfo]:
    """Get context window information for a model using litellm.

    Falls back to environment variable defaults if model info is not available.
    """
    try:
        info = litellm.get_model_info(model_name)
        max_input = info.get("max_input_tokens")
        max_output = info.get("max_output_tokens")

        # Use environment variable defaults if litellm doesn't have the info
        if max_input is None and DEFAULT_MAX_INPUT_TOKENS is not None:
            max_input = DEFAULT_MAX_INPUT_TOKENS
            logger.debug(f"Using DEFAULT_MAX_INPUT_TOKENS={max_input} for {model_name}")

        if max_output is None and DEFAULT_MAX_OUTPUT_TOKENS is not None:
            max_output = DEFAULT_MAX_OUTPUT_TOKENS
            logger.debug(f"Using DEFAULT_MAX_OUTPUT_TOKENS={max_output} for {model_name}")

        # Only return ModelInfo if we have at least one value
        if max_input is not None or max_output is not None:
            return ModelInfo(
                max_input_tokens=max_input,
                max_output_tokens=max_output
            )
        return None

    except Exception as e:
        logger.debug(f"Could not get model info for {model_name}: {e}")

        # Fall back to environment defaults if available
        if DEFAULT_MAX_INPUT_TOKENS is not None or DEFAULT_MAX_OUTPUT_TOKENS is not None:
            logger.debug(f"Using environment defaults for {model_name}")
            return ModelInfo(
                max_input_tokens=DEFAULT_MAX_INPUT_TOKENS,
                max_output_tokens=DEFAULT_MAX_OUTPUT_TOKENS
            )
        return None


def build_trace_echo_response(original_request: MessagesRequest) -> MessagesResponse:
    return MessagesResponse(
        id=f"msg_{uuid.uuid4().hex[:24]}",
        model=original_request.original_model or original_request.model,
        role="assistant",
        content=[{"type": "text", "text": "TRACE_GATEWAY_ECHO_OK"}],
        stop_reason="end_turn",
        stop_sequence=None,
        usage=Usage(
            input_tokens=estimate_trace_echo_tokens(original_request),
            output_tokens=5,
        ),
    )

@app.middleware("http")
async def log_requests(request: Request, call_next):
    # Get request details
    method = request.method
    path = request.url.path

    # Log only basic request details at debug level
    logger.debug(f"Request: {method} {path}")

    # Debug: Log all incoming requests to /v1/messages
    if path == "/v1/messages":
        logger.warning(f"🔴 MIDDLEWARE: Incoming request to {method} {path}")

    # Process the request and get the response
    response = await call_next(request)

    # Debug: Log response from /v1/messages
    if path == "/v1/messages":
        logger.warning(f"🔴 MIDDLEWARE: Response from {method} {path}, status={response.status_code}")

    return response

# Not using validation function as we're using the environment API key

def parse_tool_result_content(content):
    """Helper function to properly parse and normalize tool result content."""
    if content is None:
        return "No content provided"
        
    if isinstance(content, str):
        return content
        
    if isinstance(content, list):
        result = ""
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                result += item.get("text", "") + "\n"
            elif isinstance(item, str):
                result += item + "\n"
            elif isinstance(item, dict):
                if "text" in item:
                    result += item.get("text", "") + "\n"
                else:
                    try:
                        result += json.dumps(item) + "\n"
                    except:
                        result += str(item) + "\n"
            else:
                try:
                    result += str(item) + "\n"
                except:
                    result += "Unparseable content\n"
        return result.strip()
        
    if isinstance(content, dict):
        if content.get("type") == "text":
            return content.get("text", "")
        try:
            return json.dumps(content)
        except:
            return str(content)
            
    # Fallback for any other type
    try:
        return str(content)
    except:
        return "Unparseable content"

def convert_anthropic_to_litellm(anthropic_request: MessagesRequest) -> Dict[str, Any]:
    """Convert Anthropic API request format to LiteLLM format (which follows OpenAI)."""
    # LiteLLM already handles Anthropic models when using the format model="anthropic/claude-3-opus-20240229"
    # So we just need to convert our Pydantic model to a dict in the expected format
    
    messages = []
    
    # Add system message if present
    if anthropic_request.system:
        # Handle different formats of system messages
        if isinstance(anthropic_request.system, str):
            # Simple string format
            messages.append({"role": "system", "content": anthropic_request.system})
        elif isinstance(anthropic_request.system, list):
            # Take only the longest text block (match server2.py behavior)
            # Claude Code sends multiple system blocks; the longest one is the main prompt
            texts = []
            for block in anthropic_request.system:
                if hasattr(block, 'type') and block.type == "text":
                    texts.append(block.text)
                elif isinstance(block, dict) and block.get("type") == "text":
                    texts.append(block.get("text", ""))
            if texts:
                system_text = max(texts, key=len)
                messages.append({"role": "system", "content": system_text})
    
    # Add conversation messages — emit OpenAI-compatible tool calling format
    # so that downstream OpenAI-compatible endpoints (Qwen, etc.) can recognize
    # tool_use / tool_result blocks instead of seeing them as flattened prose.
    def _block_attr(block, attr, default=None):
        if hasattr(block, attr):
            return getattr(block, attr)
        if isinstance(block, dict):
            return block.get(attr, default)
        return default

    for idx, msg in enumerate(anthropic_request.messages):
        content = msg.content
        if isinstance(content, str):
            messages.append({"role": msg.role, "content": content})
            continue

        if msg.role == "assistant":
            # Split text vs tool_use; emit a single assistant message with
            # OpenAI-style `tool_calls`.
            text_parts = []
            tool_calls = []
            for block in content:
                btype = _block_attr(block, "type")
                if btype == "text":
                    text_parts.append(_block_attr(block, "text", "") or "")
                elif btype == "tool_use":
                    tool_input = _block_attr(block, "input", {}) or {}
                    if not isinstance(tool_input, str):
                        try:
                            arguments_str = json.dumps(tool_input, ensure_ascii=False)
                        except (TypeError, ValueError):
                            arguments_str = str(tool_input)
                    else:
                        arguments_str = tool_input
                    tool_calls.append({
                        "id": _block_attr(block, "id", "") or "",
                        "type": "function",
                        "function": {
                            "name": _block_attr(block, "name", "") or "",
                            "arguments": arguments_str,
                        },
                    })

            assistant_msg = {"role": "assistant"}
            joined_text = "\n".join(t for t in text_parts if t)
            # OpenAI requires content to be present; null is allowed when tool_calls are set.
            assistant_msg["content"] = joined_text if joined_text else (None if tool_calls else "")
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls
            messages.append(assistant_msg)
            continue

        # role == "user": tool_result blocks become standalone `tool` role
        # messages; remaining text/image stays on the user message that follows.
        user_text_parts = []
        user_image_blocks = []
        tool_messages = []
        for block in content:
            btype = _block_attr(block, "type")
            if btype == "text":
                user_text_parts.append(_block_attr(block, "text", "") or "")
            elif btype == "image":
                user_image_blocks.append({"type": "image", "source": _block_attr(block, "source", {})})
            elif btype == "tool_use":
                # Anthropic spec doesn't put tool_use on user messages, but tolerate it.
                user_text_parts.append(
                    f"[tool_use {_block_attr(block, 'name', '')}({_block_attr(block, 'id', '')})]"
                )
            elif btype == "tool_result":
                tool_id = _block_attr(block, "tool_use_id", "") or ""
                result_text = parse_tool_result_content(_block_attr(block, "content"))
                tool_messages.append({
                    "role": "tool",
                    "tool_call_id": tool_id,
                    "content": result_text,
                })

        # tool messages must come right after the assistant's tool_calls message
        # and before any follow-up user text, per OpenAI's chat schema.
        messages.extend(tool_messages)

        if user_image_blocks or user_text_parts:
            joined_text = "\n".join(t for t in user_text_parts if t)
            if user_image_blocks:
                user_content = list(user_image_blocks)
                if joined_text:
                    user_content.insert(0, {"type": "text", "text": joined_text})
                messages.append({"role": "user", "content": user_content})
            elif joined_text:
                messages.append({"role": "user", "content": joined_text})
    
    # Cap max_tokens for OpenAI models to their limit of 16384
    max_tokens = anthropic_request.max_tokens
    if anthropic_request.model.startswith("openai/") or anthropic_request.model.startswith("gemini/"):
        max_tokens = min(max_tokens, 16384)
        logger.debug(f"Capping max_tokens to 16384 for OpenAI/Gemini model (original value: {anthropic_request.max_tokens})")
    
    # Create LiteLLM request dict
    litellm_request = {
        "model": anthropic_request.model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": anthropic_request.temperature,
        "stream": anthropic_request.stream,  # Respect client's stream preference
    }

    # Pass thinking through extra_body
    if anthropic_request.thinking:
        thinking_dict = anthropic_request.thinking
        if hasattr(thinking_dict, 'dict'):
            thinking_dict = thinking_dict.dict()
        elif hasattr(thinking_dict, 'model_dump'):
            thinking_dict = thinking_dict.model_dump()

        # 仅对 Qwen3 系列模型做映射
        model_lower = anthropic_request.model.lower()
        if "qwen3" in model_lower and isinstance(thinking_dict, dict):
            if thinking_dict.get("type") == "adaptive":
                thinking_dict["type"] = "auto"

        litellm_request["extra_body"] = {"thinking": thinking_dict}

    # Add optional parameters if present
    if anthropic_request.stop_sequences:
        litellm_request["stop"] = anthropic_request.stop_sequences
    
    if anthropic_request.top_p:
        litellm_request["top_p"] = anthropic_request.top_p
    
    if anthropic_request.top_k:
        litellm_request["top_k"] = anthropic_request.top_k
    
    # Convert tools to OpenAI format
    if anthropic_request.tools:
        openai_tools = []
        is_gemini_model = anthropic_request.model.startswith("gemini/")

        for tool in anthropic_request.tools:
            # Convert to dict if it's a pydantic model
            if hasattr(tool, 'dict'):
                tool_dict = tool.dict()
            else:
                # Ensure tool_dict is a dictionary, handle potential errors if 'tool' isn't dict-like
                try:
                    tool_dict = dict(tool) if not isinstance(tool, dict) else tool
                except (TypeError, ValueError):
                     logger.error(f"Could not convert tool to dict: {tool}")
                     continue # Skip this tool if conversion fails

            # Clean the schema if targeting a Gemini model
            input_schema = tool_dict.get("input_schema", {})
            if is_gemini_model:
                 logger.debug(f"Cleaning schema for Gemini tool: {tool_dict.get('name')}")
                 input_schema = clean_gemini_schema(input_schema)

            # Create OpenAI-compatible function tool
            openai_tool = {
                "type": "function",
                "function": {
                    "name": tool_dict["name"],
                    "description": tool_dict.get("description", ""),
                    "parameters": input_schema # Use potentially cleaned schema
                }
            }
            openai_tools.append(openai_tool)

        litellm_request["tools"] = openai_tools
    
    # Convert tool_choice to OpenAI format if present
    if anthropic_request.tool_choice:
        if hasattr(anthropic_request.tool_choice, 'dict'):
            tool_choice_dict = anthropic_request.tool_choice.dict()
        else:
            tool_choice_dict = anthropic_request.tool_choice
            
        # Handle Anthropic's tool_choice format
        choice_type = tool_choice_dict.get("type")
        if choice_type == "auto":
            litellm_request["tool_choice"] = "auto"
        elif choice_type == "any":
            litellm_request["tool_choice"] = "any"
        elif choice_type == "tool" and "name" in tool_choice_dict:
            litellm_request["tool_choice"] = {
                "type": "function",
                "function": {"name": tool_choice_dict["name"]}
            }
        else:
            # Default to auto if we can't determine
            litellm_request["tool_choice"] = "auto"
    
    return litellm_request

def convert_litellm_to_anthropic(litellm_response: Union[Dict[str, Any], Any],
                                 original_request: MessagesRequest) -> MessagesResponse:
    """Convert LiteLLM (OpenAI format) response to Anthropic API response format."""

    # Enhanced response extraction with better error handling
    try:
        # Get the clean model name to check capabilities
        clean_model = original_request.model
        if clean_model.startswith("anthropic/"):
            clean_model = clean_model[len("anthropic/"):]
        elif clean_model.startswith("openai/"):
            clean_model = clean_model[len("openai/"):]
        elif clean_model.startswith("gemini/"):
            clean_model = clean_model[len("gemini/"):]

        # Get model context information for the actual model being used
        model_info = get_model_context_info(clean_model)

        # Handle ModelResponse object from LiteLLM
        if hasattr(litellm_response, 'choices') and hasattr(litellm_response, 'usage'):
            # Extract data from ModelResponse object directly
            choices = litellm_response.choices
            message = choices[0].message if choices and len(choices) > 0 else None
            content_text = message.content if message and hasattr(message, 'content') else ""
            tool_calls = message.tool_calls if message and hasattr(message, 'tool_calls') else None
            finish_reason = choices[0].finish_reason if choices and len(choices) > 0 else "stop"
            usage_info = litellm_response.usage
            response_id = getattr(litellm_response, 'id', f"msg_{uuid.uuid4()}")
        else:
            # For backward compatibility - handle dict responses
            # If response is a dict, use it, otherwise try to convert to dict
            try:
                response_dict = litellm_response if isinstance(litellm_response, dict) else litellm_response.dict()
            except AttributeError:
                # If .dict() fails, try to use model_dump or __dict__ 
                try:
                    response_dict = litellm_response.model_dump() if hasattr(litellm_response, 'model_dump') else litellm_response.__dict__
                except AttributeError:
                    # Fallback - manually extract attributes
                    response_dict = {
                        "id": getattr(litellm_response, 'id', f"msg_{uuid.uuid4()}"),
                        "choices": getattr(litellm_response, 'choices', [{}]),
                        "usage": getattr(litellm_response, 'usage', {})
                    }
                    
            # Extract the content from the response dict
            choices = response_dict.get("choices", [{}])
            message = choices[0].get("message", {}) if choices and len(choices) > 0 else {}
            content_text = message.get("content", "")
            tool_calls = message.get("tool_calls", None)
            finish_reason = choices[0].get("finish_reason", "stop") if choices and len(choices) > 0 else "stop"
            usage_info = response_dict.get("usage", {})
            response_id = response_dict.get("id", f"msg_{uuid.uuid4()}")
        
        # Create content list for Anthropic format
        content = []
        
        # Add text content block if present (text might be None or empty for pure tool call responses)
        if content_text is not None and content_text != "":
            content.append({"type": "text", "text": content_text})

        # Always convert tool_calls to Anthropic tool_use blocks — the response
        # contract is Anthropic regardless of upstream provider, so Claude Code
        # needs structured tool_use even for Qwen / OpenAI-compatible models.
        if tool_calls:
            logger.debug(f"Processing tool calls: {tool_calls}")

            # Convert to list if it's not already
            if not isinstance(tool_calls, list):
                tool_calls = [tool_calls]

            for idx, tool_call in enumerate(tool_calls):
                logger.debug(f"Processing tool call {idx}: {tool_call}")

                # Extract function data based on whether it's a dict or object
                if isinstance(tool_call, dict):
                    function = tool_call.get("function", {})
                    tool_id = tool_call.get("id", f"toolu_{uuid.uuid4().hex[:24]}")
                    name = function.get("name", "")
                    arguments = function.get("arguments", "{}")
                else:
                    function = getattr(tool_call, "function", None)
                    tool_id = getattr(tool_call, "id", f"toolu_{uuid.uuid4().hex[:24]}")
                    name = getattr(function, "name", "") if function else ""
                    arguments = getattr(function, "arguments", "{}") if function else "{}"

                # Convert string arguments to dict if needed
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments) if arguments else {}
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse tool arguments as JSON: {arguments}")
                        arguments = {"raw": arguments}
                if not isinstance(arguments, dict):
                    arguments = {"value": arguments}

                logger.debug(f"Adding tool_use block: id={tool_id}, name={name}, input={arguments}")

                content.append({
                    "type": "tool_use",
                    "id": tool_id,
                    "name": name,
                    "input": arguments
                })
        
        # Get usage information - extract values safely from object or dict
        if isinstance(usage_info, dict):
            prompt_tokens = usage_info.get("prompt_tokens", 0)
            completion_tokens = usage_info.get("completion_tokens", 0)
        else:
            prompt_tokens = getattr(usage_info, "prompt_tokens", 0)
            completion_tokens = getattr(usage_info, "completion_tokens", 0)
        
        # Map OpenAI finish_reason to Anthropic stop_reason
        stop_reason = None
        if finish_reason == "stop":
            stop_reason = "end_turn"
        elif finish_reason == "length":
            stop_reason = "max_tokens"
        elif finish_reason == "tool_calls":
            stop_reason = "tool_use"
        else:
            stop_reason = "end_turn"  # Default
        
        # Make sure content is never empty
        if not content:
            content.append({"type": "text", "text": ""})
        
        # Create Anthropic-style response
        anthropic_response = MessagesResponse(
            id=response_id,
            model=original_request.original_model or original_request.model,
            role="assistant",
            content=content,
            stop_reason=stop_reason,
            stop_sequence=None,
            usage=Usage(
                input_tokens=prompt_tokens,
                output_tokens=completion_tokens
            ),
            model_info=model_info  # Add model context information
        )
        
        return anthropic_response
        
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        error_message = f"Error converting response: {str(e)}\n\nFull traceback:\n{error_traceback}"
        logger.error(error_message)
        
        # In case of any error, create a fallback response
        return MessagesResponse(
            id=f"msg_{uuid.uuid4()}",
            model=original_request.model,
            role="assistant",
            content=[{"type": "text", "text": f"Error converting response: {str(e)}. Please check server logs."}],
            stop_reason="end_turn",
            usage=Usage(input_tokens=0, output_tokens=0),
            model_info=None
        )

async def handle_streaming(response_generator, original_request: MessagesRequest, trace_id=None, trace_start=None, litellm_request=None):
    """Handle streaming responses from LiteLLM and convert to Anthropic format."""
    accumulated_response = {
        "id": None,
        "model": original_request.original_model or original_request.model,
        "role": "assistant",
        "content": [],
        "stop_reason": None,
        "usage": {"input_tokens": 0, "output_tokens": 0}
    }

    # Get model context information for the actual model being used
    clean_model = original_request.model
    if clean_model.startswith("anthropic/"):
        clean_model = clean_model[len("anthropic/"):]
    elif clean_model.startswith("openai/"):
        clean_model = clean_model[len("openai/"):]
    elif clean_model.startswith("gemini/"):
        clean_model = clean_model[len("gemini/"):]

    model_info = get_model_context_info(clean_model)
    model_info_dict = None
    if model_info:
        model_info_dict = {
            "max_input_tokens": model_info.max_input_tokens,
            "max_output_tokens": model_info.max_output_tokens
        }

    try:
        # Send message_start event
        message_id = f"msg_{uuid.uuid4().hex[:24]}"  # Format similar to Anthropic's IDs
        accumulated_response["id"] = message_id

        message_data = {
            'type': 'message_start',
            'message': {
                'id': message_id,
                'type': 'message',
                'role': 'assistant',
                'model': original_request.original_model or original_request.model,
                'content': [],
                'stop_reason': None,
                'stop_sequence': None,
                'usage': {
                    'input_tokens': 0,
                    'cache_creation_input_tokens': 0,
                    'cache_read_input_tokens': 0,
                    'output_tokens': 0
                }
            }
        }

        # Add model_info to message_start if available
        if model_info_dict:
            message_data['message']['model_info'] = model_info_dict

        yield f"event: message_start\ndata: {json.dumps(message_data)}\n\n"
        
        # Content block index for the first text block
        yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
        
        # Send a ping to keep the connection alive (Anthropic does this)
        yield f"event: ping\ndata: {json.dumps({'type': 'ping'})}\n\n"
        
        tool_index = None
        current_tool_call = None
        tool_content = ""
        accumulated_text = ""  # Track accumulated text content
        text_sent = False  # Track if we've sent any text content
        text_block_closed = False  # Track if text block is closed
        input_tokens = 0
        output_tokens = 0
        has_sent_stop_reason = False
        last_tool_index = 0

        # Track tool calls for accumulated response
        tool_calls_map = {}  # {tool_id: {"name": str, "input": str}}
        
        # Process each chunk
        async for chunk in response_generator:
            try:

                
                # Check if this is the end of the response with usage data
                if hasattr(chunk, 'usage') and chunk.usage is not None:
                    if hasattr(chunk.usage, 'prompt_tokens'):
                        input_tokens = chunk.usage.prompt_tokens
                    if hasattr(chunk.usage, 'completion_tokens'):
                        output_tokens = chunk.usage.completion_tokens
                
                # Handle text content
                if hasattr(chunk, 'choices') and len(chunk.choices) > 0:
                    choice = chunk.choices[0]
                    
                    # Get the delta from the choice
                    if hasattr(choice, 'delta'):
                        delta = choice.delta
                    else:
                        # If no delta, try to get message
                        delta = getattr(choice, 'message', {})
                    
                    # Check for finish_reason to know when we're done
                    finish_reason = getattr(choice, 'finish_reason', None)
                    
                    # Process text content
                    delta_content = None
                    
                    # Handle different formats of delta content
                    if hasattr(delta, 'content'):
                        delta_content = delta.content
                    elif isinstance(delta, dict) and 'content' in delta:
                        delta_content = delta['content']
                    
                    # Accumulate text content
                    if delta_content is not None and delta_content != "":
                        accumulated_text += delta_content
                        
                        # Always emit text deltas if no tool calls started
                        if tool_index is None and not text_block_closed:
                            text_sent = True
                            yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': delta_content}})}\n\n"
                    
                    # Process tool calls
                    delta_tool_calls = None
                    
                    # Handle different formats of tool calls
                    if hasattr(delta, 'tool_calls'):
                        delta_tool_calls = delta.tool_calls
                    elif isinstance(delta, dict) and 'tool_calls' in delta:
                        delta_tool_calls = delta['tool_calls']
                    
                    # Process tool calls if any
                    if delta_tool_calls:
                        # First tool call we've seen - need to handle text properly
                        if tool_index is None:
                            # If we've been streaming text, close that text block
                            if text_sent and not text_block_closed:
                                text_block_closed = True
                                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
                            # If we've accumulated text but not sent it, we need to emit it now
                            # This handles the case where the first delta has both text and a tool call
                            elif accumulated_text and not text_sent and not text_block_closed:
                                # Send the accumulated text
                                text_sent = True
                                yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': accumulated_text}})}\n\n"
                                # Close the text block
                                text_block_closed = True
                                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
                            # Close text block even if we haven't sent anything - models sometimes emit empty text blocks
                            elif not text_block_closed:
                                text_block_closed = True
                                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
                                
                        # Convert to list if it's not already
                        if not isinstance(delta_tool_calls, list):
                            delta_tool_calls = [delta_tool_calls]
                        
                        for tool_call in delta_tool_calls:
                            # Get the index of this tool call (for multiple tools)
                            current_index = None
                            if isinstance(tool_call, dict) and 'index' in tool_call:
                                current_index = tool_call['index']
                            elif hasattr(tool_call, 'index'):
                                current_index = tool_call.index
                            else:
                                current_index = 0
                            
                            # Check if this is a new tool or a continuation
                            if tool_index is None or current_index != tool_index:
                                # New tool call - create a new tool_use block
                                tool_index = current_index
                                last_tool_index += 1
                                anthropic_tool_index = last_tool_index

                                # Extract function info
                                if isinstance(tool_call, dict):
                                    function = tool_call.get('function', {})
                                    name = function.get('name', '') if isinstance(function, dict) else ""
                                    tool_id = tool_call.get('id', f"toolu_{uuid.uuid4().hex[:24]}")
                                else:
                                    function = getattr(tool_call, 'function', None)
                                    name = getattr(function, 'name', '') if function else ''
                                    tool_id = getattr(tool_call, 'id', f"toolu_{uuid.uuid4().hex[:24]}")

                                # Initialize tool call tracking
                                if tool_id not in tool_calls_map:
                                    tool_calls_map[tool_id] = {"name": name, "input": ""}

                                # Start a new tool_use block
                                yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': anthropic_tool_index, 'content_block': {'type': 'tool_use', 'id': tool_id, 'name': name, 'input': {}}})}\n\n"
                                current_tool_call = tool_call
                                tool_content = ""
                            
                            # Extract function arguments
                            arguments = None
                            if isinstance(tool_call, dict) and 'function' in tool_call:
                                function = tool_call.get('function', {})
                                arguments = function.get('arguments', '') if isinstance(function, dict) else ''
                            elif hasattr(tool_call, 'function'):
                                function = getattr(tool_call, 'function', None)
                                arguments = getattr(function, 'arguments', '') if function else ''
                            
                            # If we have arguments, send them as a delta
                            if arguments:
                                # Try to detect if arguments are valid JSON or just a fragment
                                try:
                                    # If it's already a dict, use it
                                    if isinstance(arguments, dict):
                                        args_json = json.dumps(arguments)
                                    else:
                                        # Otherwise, try to parse it
                                        json.loads(arguments)
                                        args_json = arguments
                                except (json.JSONDecodeError, TypeError):
                                    # If it's a fragment, treat it as a string
                                    args_json = arguments

                                # Add to accumulated tool content
                                tool_content += args_json if isinstance(args_json, str) else ""

                                # Update tool calls map
                                if tool_id in tool_calls_map:
                                    tool_calls_map[tool_id]["input"] += args_json if isinstance(args_json, str) else ""

                                # Send the update
                                yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': anthropic_tool_index, 'delta': {'type': 'input_json_delta', 'partial_json': args_json}})}\n\n"
                    
                    # Process finish_reason - end the streaming response
                    if finish_reason and not has_sent_stop_reason:
                        has_sent_stop_reason = True
                        
                        # Close any open tool call blocks
                        if tool_index is not None:
                            for i in range(1, last_tool_index + 1):
                                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': i})}\n\n"
                        
                        # If we accumulated text but never sent or closed text block, do it now
                        if not text_block_closed:
                            if accumulated_text and not text_sent:
                                # Send the accumulated text
                                yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': accumulated_text}})}\n\n"
                            # Close the text block
                            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
                        
                        # Map OpenAI finish_reason to Anthropic stop_reason
                        stop_reason = "end_turn"
                        if finish_reason == "length":
                            stop_reason = "max_tokens"
                        elif finish_reason == "tool_calls":
                            stop_reason = "tool_use"
                        elif finish_reason == "stop":
                            stop_reason = "end_turn"

                        # Update accumulated response
                        accumulated_response["stop_reason"] = stop_reason
                        accumulated_response["usage"]["input_tokens"] = input_tokens
                        accumulated_response["usage"]["output_tokens"] = output_tokens
                        if accumulated_text:
                            accumulated_response["content"].append({"type": "text", "text": accumulated_text})

                        # Add tool calls to accumulated response
                        for tool_id, tool_data in tool_calls_map.items():
                            try:
                                tool_input = json.loads(tool_data["input"]) if tool_data["input"] else {}
                            except json.JSONDecodeError:
                                tool_input = {"raw": tool_data["input"]}
                            accumulated_response["content"].append({
                                "type": "tool_use",
                                "id": tool_id,
                                "name": tool_data["name"],
                                "input": tool_input
                            })

                        # Send message_delta with stop reason and usage
                        usage = {"output_tokens": output_tokens}

                        yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': stop_reason, 'stop_sequence': None}, 'usage': usage})}\n\n"

                        # Send message_stop event
                        yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"

                        # Send final [DONE] marker to match Anthropic's behavior
                        yield "data: [DONE]\n\n"

                        # Record trace completion (after streaming finishes)
                        if trace_id and trace_start and litellm_request:
                            try:
                                trace_request_completed(
                                    trace_id=trace_id,
                                    status_code=200,
                                    duration_ms=monotonic_ms(trace_start),
                                    converted_request=litellm_request,
                                    response=accumulated_response,
                                    extra={"streaming": True},
                                )
                            except Exception as trace_err:
                                logger.error(f"Failed to record streaming trace: {trace_err}")

                        return
            except Exception as e:
                # Log error but continue processing other chunks
                logger.error(f"Error processing chunk: {str(e)}")
                continue
        
        # If we didn't get a finish reason, close any open blocks
        if not has_sent_stop_reason:
            # Close any open tool call blocks
            if tool_index is not None:
                for i in range(1, last_tool_index + 1):
                    yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': i})}\n\n"

            # Close the text content block
            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"

            # Send final message_delta with usage
            usage = {"output_tokens": output_tokens}

            # Update accumulated response
            accumulated_response["stop_reason"] = "end_turn"
            accumulated_response["usage"]["input_tokens"] = input_tokens
            accumulated_response["usage"]["output_tokens"] = output_tokens
            if accumulated_text:
                accumulated_response["content"].append({"type": "text", "text": accumulated_text})

            # Add tool calls to accumulated response
            for tool_id, tool_data in tool_calls_map.items():
                try:
                    tool_input = json.loads(tool_data["input"]) if tool_data["input"] else {}
                except json.JSONDecodeError:
                    tool_input = {"raw": tool_data["input"]}
                accumulated_response["content"].append({
                    "type": "tool_use",
                    "id": tool_id,
                    "name": tool_data["name"],
                    "input": tool_input
                })

            yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'end_turn', 'stop_sequence': None}, 'usage': usage})}\n\n"

            # Send message_stop event
            yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"

            # Send final [DONE] marker to match Anthropic's behavior
            yield "data: [DONE]\n\n"

            # Record trace completion
            if trace_id and trace_start and litellm_request:
                try:
                    trace_request_completed(
                        trace_id=trace_id,
                        status_code=200,
                        duration_ms=monotonic_ms(trace_start),
                        converted_request=litellm_request,
                        response=accumulated_response,
                        extra={"streaming": True},
                    )
                except Exception as trace_err:
                    logger.error(f"Failed to record streaming trace: {trace_err}")

    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        error_message = f"Error in streaming: {str(e)}\n\nFull traceback:\n{error_traceback}"
        logger.error(error_message)
        
        # Send error message_delta
        yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'error', 'stop_sequence': None}, 'usage': {'output_tokens': 0}})}\n\n"
        
        # Send message_stop event
        yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
        
        # Send final [DONE] marker
        yield "data: [DONE]\n\n"

@app.post("/v1/messages")
async def create_message(
    request: MessagesRequest,
    raw_request: Request
):
    trace_id = new_trace_id("msg")
    trace_start = time.time()
    litellm_request = None
    body_json: Dict[str, Any] = {}

    # Debug: Log entry to this endpoint
    logger.warning(f"🔵 ENDPOINT ENTRY: /v1/messages - trace_id={trace_id}")

    try:
        # print the body here
        body = await raw_request.body()

        # Parse the raw body as JSON since it's bytes
        body_json = json.loads(body.decode('utf-8'))
        original_model = body_json.get("model", "unknown")
        trace_request_started(
            trace_id=trace_id,
            api="messages",
            method=raw_request.method,
            path=raw_request.url.path,
            headers=headers_to_trace(raw_request.headers.items()),
            client=raw_request.client.host if raw_request.client else None,
            body_json=body_json,
            mapped_model=request.model,
        )
        
        # Get the display name for logging, just the model name without provider prefix
        display_model = original_model
        if "/" in display_model:
            display_model = display_model.split("/")[-1]
        
        # Clean model name for capability check
        clean_model = request.model
        if clean_model.startswith("anthropic/"):
            clean_model = clean_model[len("anthropic/"):]
        elif clean_model.startswith("openai/"):
            clean_model = clean_model[len("openai/"):]

        logger.debug(f"📊 PROCESSING REQUEST: Model={request.model}, Stream={request.stream}")

        # Anthropic passthrough: when the upstream is an Anthropic-compatible
        # API (official Anthropic, DeepSeek's Anthropic endpoint, yunwu.ai,
        # etc.), bypass LiteLLM's OpenAI-format conversion and forward the
        # client's raw Anthropic request. This preserves native semantics
        # (thinking blocks, tool_use/tool_result, server tools) that LiteLLM
        # doesn't always round-trip correctly.
        if request.model.startswith("anthropic/"):
            logger.debug(
                f"Using Anthropic passthrough to {ANTHROPIC_BASE_URL} for model: {request.model}"
            )
            return await _handle_anthropic_passthrough(
                request=request,
                raw_request=raw_request,
                body_json=body_json,
                trace_id=trace_id,
                trace_start=trace_start,
                display_model=display_model,
            )

        # Convert Anthropic request to LiteLLM format
        litellm_request = convert_anthropic_to_litellm(request)

        # Determine which API key to use based on the model
        if request.model.startswith("openai/"):
            litellm_request["api_key"] = OPENAI_API_KEY
            if OPENAI_BASE_URL:
                litellm_request["api_base"] = OPENAI_BASE_URL
                logger.debug(f"Using OpenAI API key and custom base URL {OPENAI_BASE_URL} for model: {request.model}")
            else:
                logger.debug(f"Using OpenAI API key for model: {request.model}")
        elif request.model.startswith("gemini/"):
            if USE_VERTEX_AUTH:
                litellm_request["vertex_project"] = VERTEX_PROJECT
                litellm_request["vertex_location"] = VERTEX_LOCATION
                litellm_request["custom_llm_provider"] = "vertex_ai"
                logger.debug(f"Using Gemini ADC with project={VERTEX_PROJECT}, location={VERTEX_LOCATION} and model: {request.model}")
            else:
                litellm_request["api_key"] = GEMINI_API_KEY
                logger.debug(f"Using Gemini API key for model: {request.model}")
        else:
            litellm_request["api_key"] = ANTHROPIC_API_KEY
            logger.debug(f"Using Anthropic API key for model: {request.model}")

        # For OpenAI-compatible models — sanitize message content while preserving
        # OpenAI-native tool calling structure (tool_calls + role:"tool"), which
        # Qwen and other OpenAI-compatible providers expect.
        if "openai" in litellm_request["model"] and "messages" in litellm_request:
            logger.debug(f"Processing OpenAI model request: {litellm_request['model']}")

            allowed_keys = {"role", "content", "name", "tool_call_id", "tool_calls"}

            for i, msg in enumerate(litellm_request["messages"]):
                role = msg.get("role")
                content = msg.get("content")

                # tool messages already carry plain string content from the converter — leave alone
                if role == "tool":
                    if not isinstance(content, str):
                        msg["content"] = parse_tool_result_content(content)
                # assistant messages with tool_calls: keep tool_calls; ensure content is str-or-None
                elif role == "assistant" and msg.get("tool_calls"):
                    if isinstance(content, list):
                        text_parts = []
                        for block in content:
                            if isinstance(block, dict) and block.get("type") == "text":
                                text_parts.append(block.get("text", ""))
                        msg["content"] = "\n".join(p for p in text_parts if p) or None
                    elif content == "":
                        msg["content"] = None
                else:
                    # Plain user/assistant/system messages: collapse list content to string,
                    # preserving any tool_use / tool_result that slipped through (defensive).
                    if isinstance(content, list):
                        text_content = ""
                        for block in content:
                            if not isinstance(block, dict):
                                continue
                            btype = block.get("type")
                            if btype == "text":
                                text_content += block.get("text", "") + "\n"
                            elif btype == "tool_result":
                                tool_id = block.get("tool_use_id", "unknown")
                                text_content += f"[Tool Result ID: {tool_id}]\n"
                                text_content += parse_tool_result_content(block.get("content")) + "\n"
                            elif btype == "tool_use":
                                tool_name = block.get("name", "unknown")
                                tool_id = block.get("id", "unknown")
                                tool_input = json.dumps(block.get("input", {}), ensure_ascii=False)
                                text_content += f"[Tool: {tool_name} (ID: {tool_id})]\nInput: {tool_input}\n\n"
                            elif btype == "image":
                                text_content += "[Image content - not displayed in text format]\n"
                        msg["content"] = text_content.strip() or "..."
                    elif content is None:
                        msg["content"] = "..."

                # Strip any keys the OpenAI schema doesn't recognize on a message
                for key in list(msg.keys()):
                    if key not in allowed_keys:
                        logger.warning(f"Removing unsupported field from message: {key}")
                        del msg[key]

            # Final defensive sweep
            for i, msg in enumerate(litellm_request["messages"]):
                logger.debug(f"Message {i} format check - role: {msg.get('role')}, content type: {type(msg.get('content'))}")
                if isinstance(msg.get("content"), list):
                    logger.warning(f"CRITICAL: Message {i} still has list content after processing")
                    msg["content"] = json.dumps(msg.get("content"), ensure_ascii=False)
                elif msg.get("content") is None and not msg.get("tool_calls"):
                    logger.warning(f"Message {i} has None content - replacing with placeholder")
                    msg["content"] = "..."
        
        if TRACE_ECHO_ONLY:
            logger.warning("CC_TRACE_ECHO_ONLY enabled; returning deterministic trace echo response without upstream call")
            num_tools = len(request.tools) if request.tools else 0
            log_request_beautifully(
                "POST",
                raw_request.url.path,
                display_model,
                litellm_request.get('model'),
                len(litellm_request['messages']),
                num_tools,
                200
            )
            anthropic_response = build_trace_echo_response(request)
            trace_request_completed(
                trace_id=trace_id,
                status_code=200,
                duration_ms=monotonic_ms(trace_start),
                converted_request=litellm_request,
                response=anthropic_response,
                extra={"echo_only": True},
            )
            return anthropic_response
        
        # Only log basic info about the request, not the full details
        logger.debug(f"Request for model: {litellm_request.get('model')}, stream: {litellm_request.get('stream', False)}")

        # Handle streaming vs non-streaming based on client request
        num_tools = len(request.tools) if request.tools else 0

        log_request_beautifully(
            "POST",
            raw_request.url.path,
            display_model,
            litellm_request.get('model'),
            len(litellm_request['messages']),
            num_tools,
            200
        )
        start_time = time.time()

        # Check if client requested streaming
        if request.stream:
            logger.warning(f"🟡 STREAMING MODE: trace_id={trace_id}, model={litellm_request.get('model')}")

            # Retry transient upstream errors (rate limits, 5xx, timeouts) with
            # exponential backoff. Configurable via LITELLM_MAX_RETRIES env.
            max_retries = int(os.environ.get("LITELLM_MAX_RETRIES", "3"))
            base_delay = float(os.environ.get("LITELLM_RETRY_BASE_DELAY", "2"))
            litellm_response = None
            last_exc = None
            for attempt in range(max_retries + 1):
                try:
                    # Debug: Log before making upstream call
                    logger.warning(f"🟢 UPSTREAM STREAMING CALL START: trace_id={trace_id}, attempt={attempt+1}/{max_retries+1}, model={litellm_request.get('model')}")

                    # Async call so a slow upstream / retry sleep doesn't block
                    # other concurrent requests on the event loop.
                    litellm_response = await litellm.acompletion(**litellm_request)

                    # Debug: Log after successful upstream call
                    logger.warning(f"🟢 UPSTREAM STREAMING CALL SUCCESS: trace_id={trace_id}, attempt={attempt+1}")
                    break
                except Exception as exc:
                    last_exc = exc
                    status_code = getattr(exc, "status_code", None)
                    exc_name = type(exc).__name__
                    is_transient = (
                        isinstance(exc, getattr(litellm, "RateLimitError", tuple()))
                        or isinstance(exc, getattr(litellm, "Timeout", tuple()))
                        or isinstance(exc, getattr(litellm, "APIConnectionError", tuple()))
                        or isinstance(exc, getattr(litellm, "InternalServerError", tuple()))
                        or isinstance(exc, getattr(litellm, "ServiceUnavailableError", tuple()))
                        or status_code in (408, 425, 429, 500, 502, 503, 504)
                    )
                    if not is_transient or attempt >= max_retries:
                        raise
                    delay = base_delay * (2 ** attempt)
                    logger.warning(
                        f"⏳ Transient error ({exc_name}, status={status_code}) "
                        f"on attempt {attempt + 1}/{max_retries + 1}; "
                        f"retrying in {delay:.1f}s"
                    )
                    await asyncio.sleep(delay)

            # Return streaming response
            return StreamingResponse(
                handle_streaming(litellm_response, request, trace_id, trace_start, litellm_request),
                media_type="text/event-stream"
            )

        # Non-streaming mode
        logger.warning(f"🟡 NON-STREAMING MODE: trace_id={trace_id}, model={litellm_request.get('model')}")

        # Retry transient upstream errors (rate limits, 5xx, timeouts) with
        # exponential backoff. Configurable via LITELLM_MAX_RETRIES env.
        # Non-streaming mode
        logger.warning(f"🟡 NON-STREAMING MODE: trace_id={trace_id}, model={litellm_request.get('model')}")

        # Retry transient upstream errors (rate limits, 5xx, timeouts) with
        # exponential backoff. Configurable via LITELLM_MAX_RETRIES env.
        max_retries = int(os.environ.get("LITELLM_MAX_RETRIES", "3"))
        base_delay = float(os.environ.get("LITELLM_RETRY_BASE_DELAY", "2"))
        litellm_response = None
        last_exc = None
        for attempt in range(max_retries + 1):
            try:
                # Debug: Log before making upstream call
                logger.warning(f"🟢 UPSTREAM CALL START: trace_id={trace_id}, attempt={attempt+1}/{max_retries+1}, model={litellm_request.get('model')}")

                # Async call so a slow upstream / retry sleep doesn't block
                # other concurrent requests on the event loop.
                litellm_response = await litellm.acompletion(**litellm_request)

                # Debug: Log after successful upstream call
                logger.warning(f"🟢 UPSTREAM CALL SUCCESS: trace_id={trace_id}, attempt={attempt+1}")
                break
            except Exception as exc:
                last_exc = exc
                status_code = getattr(exc, "status_code", None)
                exc_name = type(exc).__name__
                is_transient = (
                    isinstance(exc, getattr(litellm, "RateLimitError", tuple()))
                    or isinstance(exc, getattr(litellm, "Timeout", tuple()))
                    or isinstance(exc, getattr(litellm, "APIConnectionError", tuple()))
                    or isinstance(exc, getattr(litellm, "InternalServerError", tuple()))
                    or isinstance(exc, getattr(litellm, "ServiceUnavailableError", tuple()))
                    or status_code in (408, 425, 429, 500, 502, 503, 504)
                )
                if not is_transient or attempt >= max_retries:
                    raise
                delay = base_delay * (2 ** attempt)
                logger.warning(
                    f"⏳ Transient error ({exc_name}, status={status_code}) "
                    f"on attempt {attempt + 1}/{max_retries + 1}; "
                    f"retrying in {delay:.1f}s"
                )
                await asyncio.sleep(delay)
        logger.debug(f"✅ RESPONSE RECEIVED: Model={litellm_request.get('model')}, Time={time.time() - start_time:.2f}s")

        # Convert LiteLLM response to Anthropic format
        anthropic_response = convert_litellm_to_anthropic(litellm_response, request)

        trace_request_completed(
            trace_id=trace_id,
            status_code=200,
            duration_ms=monotonic_ms(trace_start),
            converted_request=litellm_request,
            response=anthropic_response,
            extra={"upstream_response": model_to_plain(litellm_response)},
        )

        return anthropic_response
                
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        
        # Capture as much info as possible about the error
        error_details = {
            "error": str(e),
            "type": type(e).__name__,
            "traceback": error_traceback
        }
        
        # Check for LiteLLM-specific attributes
        for attr in ['message', 'status_code', 'response', 'llm_provider', 'model']:
            if hasattr(e, attr):
                error_details[attr] = getattr(e, attr)
        
        # Check for additional exception details in dictionaries
        if hasattr(e, '__dict__'):
            for key, value in e.__dict__.items():
                if key not in error_details and key not in ['args', '__traceback__']:
                    error_details[key] = str(value)
        
        # Helper function to safely serialize objects for JSON
        def sanitize_for_json(obj):
            """递归地清理对象使其可以JSON序列化"""
            if isinstance(obj, dict):
                return {k: sanitize_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [sanitize_for_json(item) for item in obj]
            elif hasattr(obj, '__dict__'):
                return sanitize_for_json(obj.__dict__)
            elif hasattr(obj, 'text'):
                return str(obj.text)
            else:
                try:
                    json.dumps(obj)
                    return obj
                except (TypeError, ValueError):
                    return str(obj)
        
        # Log all error details with safe serialization
        sanitized_details = sanitize_for_json(error_details)
        logger.error(f"Error processing request: {json.dumps(sanitized_details, indent=2)}")
        
        # Format error for response
        error_message = f"Error: {str(e)}"
        if 'message' in error_details and error_details['message']:
            error_message += f"\nMessage: {error_details['message']}"
        if 'response' in error_details and error_details['response']:
            error_message += f"\nResponse: {error_details['response']}"
        
        # Return detailed error
        status_code = error_details.get('status_code', 500)
        trace_request_failed(
            trace_id=trace_id,
            status_code=status_code,
            duration_ms=monotonic_ms(trace_start),
            converted_request=litellm_request,
            error=sanitized_details,
        )
        raise HTTPException(status_code=status_code, detail=error_message)


# Headers we never want to forward to the upstream Anthropic API.
_ANTHROPIC_PASSTHROUGH_DROP_HEADERS = {
    "host",
    "content-length",
    "connection",
    "accept-encoding",
    "transfer-encoding",
    "x-forwarded-for",
    "x-forwarded-host",
    "x-forwarded-proto",
    "x-real-ip",
    "x-api-key",
    "authorization",
    "anthropic-version",
}


def _build_anthropic_passthrough_headers(raw_request: Request) -> Dict[str, str]:
    """Build headers for the upstream call. Forward client headers but
    override auth/version/host so the proxy authenticates as itself."""
    forwarded: Dict[str, str] = {}
    for k, v in raw_request.headers.items():
        if k.lower() in _ANTHROPIC_PASSTHROUGH_DROP_HEADERS:
            continue
        forwarded[k] = v

    forwarded["content-type"] = "application/json"
    forwarded["accept"] = forwarded.get("accept", "application/json")
    forwarded["anthropic-version"] = raw_request.headers.get(
        "anthropic-version", "2023-06-01"
    )
    if ANTHROPIC_API_KEY:
        forwarded["x-api-key"] = ANTHROPIC_API_KEY
    return forwarded


async def _stream_anthropic_passthrough(
    body: bytes,
    headers: Dict[str, str],
    url: str,
    trace_id: str,
    trace_start: float,
    body_json: Dict[str, Any],
):
    """Async generator: stream raw SSE bytes from upstream Anthropic API."""
    accumulated_text_chunks: List[str] = []
    upstream_status = 200
    sse_buffer = ""  # cross-chunk buffer for SSE parsing

    # Reconstruct the final response from SSE events for trace recording
    reconstructed_response: Dict[str, Any] = {
        "id": None,
        "type": "message",
        "role": "assistant",
        "content": [],
        "model": None,
        "stop_reason": None,
        "stop_sequence": None,
        "usage": {"input_tokens": 0, "output_tokens": 0}
    }

    try:
        timeout = httpx.Timeout(connect=10.0, read=600.0, write=60.0, pool=10.0)
        async with httpx.AsyncClient(timeout=timeout, verify=False) as client:
            async with client.stream("POST", url, content=body, headers=headers) as resp:
                upstream_status = resp.status_code
                if resp.status_code >= 400:
                    err_body = await resp.aread()
                    err_text = err_body.decode("utf-8", errors="replace")
                    logger.error(
                        f"Anthropic passthrough upstream error {resp.status_code}: {err_text}"
                    )
                    # Surface the upstream error as an SSE error event so the
                    # client sees a real failure instead of a hung stream.
                    yield (
                        f"event: error\ndata: "
                        f"{json.dumps({'type': 'error', 'error': {'type': 'upstream_error', 'message': err_text}})}\n\n"
                    ).encode("utf-8")
                    yield b"data: [DONE]\n\n"
                    return
                async for chunk in resp.aiter_bytes():
                    if chunk:
                        chunk_text = chunk.decode("utf-8", errors="replace")
                        accumulated_text_chunks.append(chunk_text)

                        # Parse SSE events with a cross-chunk buffer. Events are
                        # separated by a blank line ("\n\n"); a single event may
                        # span multiple network chunks, so we cannot split each
                        # chunk in isolation.
                        sse_buffer += chunk_text
                        while True:
                            sep = sse_buffer.find("\n\n")
                            if sep < 0:
                                break
                            raw_event = sse_buffer[:sep]
                            sse_buffer = sse_buffer[sep + 2:]
                            data_lines = []
                            for ln in raw_event.split("\n"):
                                if ln.startswith("data:"):
                                    # SSE spec allows "data:foo" or "data: foo"
                                    payload = ln[5:]
                                    if payload.startswith(" "):
                                        payload = payload[1:]
                                    data_lines.append(payload)
                            if not data_lines:
                                continue
                            data_str = "\n".join(data_lines).strip()
                            if not data_str or data_str == "[DONE]":
                                continue
                            try:
                                event = json.loads(data_str)
                            except json.JSONDecodeError:
                                continue
                            event_type = event.get("type")

                            if event_type == "message_start":
                                msg = event.get("message", {})
                                reconstructed_response["id"] = msg.get("id")
                                reconstructed_response["model"] = msg.get("model")
                                reconstructed_response["usage"] = msg.get("usage", reconstructed_response["usage"])

                            elif event_type == "content_block_start":
                                content_block = event.get("content_block", {})
                                # For tool_use blocks, initialize input as empty string
                                # since we'll accumulate JSON deltas
                                if content_block.get("type") == "tool_use" and "input" in content_block:
                                    content_block["input"] = ""
                                reconstructed_response["content"].append(content_block)

                            elif event_type == "content_block_delta":
                                delta = event.get("delta", {})
                                index = event.get("index", 0)
                                if index < len(reconstructed_response["content"]):
                                    block = reconstructed_response["content"][index]
                                    if delta.get("type") == "text_delta":
                                        if "text" not in block:
                                            block["text"] = ""
                                        block["text"] += delta.get("text", "")
                                    elif delta.get("type") == "thinking_delta":
                                        if "thinking" not in block:
                                            block["thinking"] = ""
                                        block["thinking"] += delta.get("thinking", "")
                                    elif delta.get("type") == "input_json_delta":
                                        if "input" not in block:
                                            block["input"] = ""
                                        elif not isinstance(block["input"], str):
                                            block["input"] = ""
                                        block["input"] += delta.get("partial_json", "")

                            elif event_type == "message_delta":
                                delta = event.get("delta", {})
                                if "stop_reason" in delta:
                                    reconstructed_response["stop_reason"] = delta["stop_reason"]
                                if "stop_sequence" in delta:
                                    reconstructed_response["stop_sequence"] = delta["stop_sequence"]
                                usage = event.get("usage", {})
                                if usage:
                                    reconstructed_response["usage"]["output_tokens"] = usage.get("output_tokens", 0)

                        yield chunk
    except Exception as exc:
        logger.error(f"Anthropic passthrough streaming failed: {exc}")
        yield (
            f"event: error\ndata: "
            f"{json.dumps({'type': 'error', 'error': {'type': 'proxy_error', 'message': str(exc)}})}\n\n"
        ).encode("utf-8")
        yield b"data: [DONE]\n\n"
        upstream_status = 500
    finally:
        try:
            # DEBUG: dump raw SSE for the latest passthrough request so we can
            # see exactly what the upstream returned. Remove once root-caused.
            try:
                from pathlib import Path as _P
                _dump = _P(__file__).parent / "cc_traces" / "last_sse.txt"
                _dump.parent.mkdir(parents=True, exist_ok=True)
                _dump.write_text(
                    f"trace_id={trace_id}\nupstream_status={upstream_status}\n"
                    f"chunks={len(accumulated_text_chunks)}\n"
                    f"=== RAW SSE ===\n" + "".join(accumulated_text_chunks),
                    encoding="utf-8",
                )
            except Exception as _dump_err:
                logger.error(f"SSE dump failed: {_dump_err}")

            # Parse tool_use blocks' input from accumulated JSON strings
            for block in reconstructed_response.get("content", []):
                if block.get("type") == "tool_use" and isinstance(block.get("input"), str):
                    try:
                        block["input"] = json.loads(block["input"])
                    except json.JSONDecodeError:
                        pass

            trace_request_completed(
                trace_id=trace_id,
                status_code=upstream_status if upstream_status < 400 else 200,
                duration_ms=monotonic_ms(trace_start),
                converted_request={"passthrough": True, "body": body_json},
                response=reconstructed_response,
                extra={"streaming": True, "passthrough": True},
            )
        except Exception as trace_err:
            logger.error(f"Failed to record passthrough streaming trace: {trace_err}")


async def _handle_anthropic_passthrough(
    request: MessagesRequest,
    raw_request: Request,
    body_json: Dict[str, Any],
    trace_id: str,
    trace_start: float,
    display_model: str,
):
    """Forward the raw client request to the Anthropic-compatible upstream
    without going through LiteLLM, preserving native Anthropic semantics
    (thinking blocks, tool_use/tool_result, server-side tools, etc.)."""
    # Strip the "anthropic/" prefix LiteLLM uses; the upstream API expects
    # the bare model name.
    upstream_model = request.model
    if upstream_model.startswith("anthropic/"):
        upstream_model = upstream_model[len("anthropic/"):]

    upstream_body = dict(body_json)
    upstream_body["model"] = upstream_model

    # Extract system messages from messages array (Claude Code v2.1.220 sends them inline)
    raw_messages = upstream_body.get("messages", [])
    if raw_messages:
        sys_contents = []
        non_sys = []
        for m in raw_messages:
            if isinstance(m, dict) and m.get("role") == "system":
                content = m.get("content", "")
                if isinstance(content, str):
                    sys_contents.append({"type": "text", "text": content})
                elif isinstance(content, list):
                    sys_contents.extend(content)
            else:
                non_sys.append(m)
        if sys_contents:
            existing_system = upstream_body.get("system")
            if existing_system is None:
                upstream_body["system"] = sys_contents
            elif isinstance(existing_system, str):
                upstream_body["system"] = [{"type": "text", "text": existing_system}] + sys_contents
            elif isinstance(existing_system, list):
                upstream_body["system"] = list(existing_system) + sys_contents
            upstream_body["messages"] = non_sys

    upstream_payload = json.dumps(upstream_body, ensure_ascii=False).encode("utf-8")

    headers = _build_anthropic_passthrough_headers(raw_request)
    url = f"{ANTHROPIC_BASE_URL}/v1/messages"

    num_tools = len(request.tools) if request.tools else 0
    log_request_beautifully(
        "POST",
        raw_request.url.path,
        display_model,
        request.model,
        len(body_json.get("messages", []) or []),
        num_tools,
        200,
    )

    if request.stream:
        logger.warning(
            f"🟡 STREAMING MODE (passthrough): trace_id={trace_id}, model={request.model}"
        )
        return StreamingResponse(
            _stream_anthropic_passthrough(
                upstream_payload, headers, url, trace_id, trace_start, body_json
            ),
            media_type="text/event-stream",
        )

    logger.warning(
        f"🟡 NON-STREAMING MODE (passthrough): trace_id={trace_id}, model={request.model}"
    )
    timeout = httpx.Timeout(connect=10.0, read=600.0, write=60.0, pool=10.0)
    async with httpx.AsyncClient(timeout=timeout, verify=False) as client:
        upstream_resp = await client.post(url, content=upstream_payload, headers=headers)

    if upstream_resp.status_code >= 400:
        err_text = upstream_resp.text
        logger.error(
            f"Anthropic passthrough upstream error {upstream_resp.status_code}: {err_text}"
        )
        try:
            err_payload = upstream_resp.json()
        except Exception:
            err_payload = {"error": err_text}
        trace_request_failed(
            trace_id=trace_id,
            status_code=upstream_resp.status_code,
            duration_ms=monotonic_ms(trace_start),
            converted_request={"passthrough": True, "body": body_json},
            error={"upstream_error": err_payload},
        )
        raise HTTPException(status_code=upstream_resp.status_code, detail=err_text)

    response_json = upstream_resp.json()
    trace_request_completed(
        trace_id=trace_id,
        status_code=200,
        duration_ms=monotonic_ms(trace_start),
        converted_request={"passthrough": True, "body": body_json},
        response=response_json,
        extra={"upstream_response": response_json, "passthrough": True},
    )
    return JSONResponse(content=response_json, status_code=200)


@app.post("/v1/messages/count_tokens")
async def count_tokens(
    request: TokenCountRequest,
    raw_request: Request
):
    trace_id = new_trace_id("tok")
    trace_start = time.time()
    converted_request = None
    request_body = model_to_plain(request)
    trace_request_started(
        trace_id=trace_id,
        api="count_tokens",
        method=raw_request.method,
        path=raw_request.url.path,
        headers=headers_to_trace(raw_request.headers.items()),
        client=raw_request.client.host if raw_request.client else None,
        body_json=request_body,
        mapped_model=request.model,
    )
    try:
        # Log the incoming token count request
        original_model = request.original_model or request.model
        
        # Get the display name for logging, just the model name without provider prefix
        display_model = original_model
        if "/" in display_model:
            display_model = display_model.split("/")[-1]
        
        # Clean model name for capability check
        clean_model = request.model
        if clean_model.startswith("anthropic/"):
            clean_model = clean_model[len("anthropic/"):]
        elif clean_model.startswith("openai/"):
            clean_model = clean_model[len("openai/"):]
        
        # Convert the messages to a format LiteLLM can understand
        converted_request = convert_anthropic_to_litellm(
            MessagesRequest(
                model=request.model,
                max_tokens=100,  # Arbitrary value not used for token counting
                messages=request.messages,
                system=request.system,
                tools=request.tools,
                tool_choice=request.tool_choice,
                thinking=request.thinking
            )
        )
        
        # Use LiteLLM's token_counter function
        try:
            # Import token_counter function
            from litellm import token_counter
            
            # Log the request beautifully
            num_tools = len(request.tools) if request.tools else 0
            
            log_request_beautifully(
                "POST",
                raw_request.url.path,
                display_model,
                converted_request.get('model'),
                len(converted_request['messages']),
                num_tools,
                200  # Assuming success at this point
            )
            
            # Prepare token counter arguments — token_counter is a local
            # tokenizer call, so no api_base / api_key is accepted or needed.
            token_counter_args = {
                "model": converted_request["model"],
                "messages": converted_request["messages"],
            }

            # Count tokens off the event loop — tokenization is CPU-bound and
            # would otherwise block other concurrent requests.
            token_count = await asyncio.to_thread(token_counter, **token_counter_args)
            
            # Return Anthropic-style response
            response = TokenCountResponse(input_tokens=token_count)
            trace_request_completed(
                trace_id=trace_id,
                status_code=200,
                duration_ms=monotonic_ms(trace_start),
                converted_request=converted_request,
                response=response,
            )
            return response
            
        except ImportError:
            logger.error("Could not import token_counter from litellm")
            # Fallback to a simple approximation
            response = TokenCountResponse(input_tokens=1000)
            trace_request_completed(
                trace_id=trace_id,
                status_code=200,
                duration_ms=monotonic_ms(trace_start),
                converted_request=converted_request,
                response=response,
                extra={"fallback": "missing_litellm_token_counter"},
            )
            return response  # Default fallback
            
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        logger.error(f"Error counting tokens: {str(e)}\n{error_traceback}")
        trace_request_failed(
            trace_id=trace_id,
            status_code=500,
            duration_ms=monotonic_ms(trace_start),
            converted_request=converted_request,
            error={"error": str(e), "type": type(e).__name__, "traceback": error_traceback},
        )
        raise HTTPException(status_code=500, detail=f"Error counting tokens: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Anthropic Proxy for LiteLLM", "trace_ui": "/trace"}

@app.get("/trace", include_in_schema=False)
async def trace_ui():
    return FileResponse(Path(__file__).parent / "static" / "trace.html")

@app.get("/api/v2/stats")
async def api_stats():
    return snapshot_stats()

@app.get("/api/v2/sessions")
async def api_list_sessions(include_archived: bool = False):
    return {"sessions": list_sessions(include_archived=include_archived), "stats": snapshot_stats()}

@app.get("/api/v2/sessions/{session_id}")
async def api_get_session(session_id: str):
    sess = get_session(session_id)
    if sess is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return sess


@app.post("/api/v2/sessions/{session_id}/archive")
async def api_archive_session(session_id: str):
    ok = archive_session(session_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"ok": True, "session_id": session_id, "archived": True}


@app.post("/api/v2/sessions/{session_id}/unarchive")
async def api_unarchive_session(session_id: str):
    ok = unarchive_session(session_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"ok": True, "session_id": session_id, "archived": False}

@app.get("/api/v2/sessions/{session_id}/timeline")
async def api_session_timeline(session_id: str):
    return {"session_id": session_id, "events": build_timeline(session_id)}

@app.get("/api/v2/sessions/{session_id}/history")
async def api_session_history(session_id: str):
    return {"session_id": session_id, "roots": build_history_chains(session_id)}

@app.get("/api/v2/sessions/{session_id}/agents")
async def api_session_agents(session_id: str):
    return {"session_id": session_id, "roots": build_agent_tree(session_id)}

@app.get("/api/v2/sessions/{session_id}/prefix_trie")
async def api_session_prefix_trie(session_id: str):
    return {"session_id": session_id, "roots": build_prefix_trie(session_id)}

@app.get("/api/v2/sessions/{session_id}/time_trajectory")
async def api_session_time_trajectory(session_id: str):
    data = build_time_trajectory(session_id)
    return {"session_id": session_id, **data}

@app.get("/api/v2/requests")
async def api_list_requests(session_id: Optional[str] = None,
                            role_kind: Optional[str] = None,
                            api: Optional[str] = None):
    return {
        "requests": list_requests(
            {"session_id": session_id, "role_kind": role_kind, "api": api}
        )
    }

@app.get("/api/v2/requests/{trace_id}")
async def api_get_request(trace_id: str):
    detail = get_request(trace_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="Trace request not found")
    return detail

@app.get("/api/v2/tool_events/{event_id}")
async def api_get_tool_event(event_id: str):
    e = get_tool_event(event_id)
    if e is None:
        raise HTTPException(status_code=404, detail="Tool event not found")
    return e

@app.get("/api/v2/agent_calls/{agent_call_id}")
async def api_get_agent_call(agent_call_id: str):
    call = get_agent_call(agent_call_id)
    if call is None:
        raise HTTPException(status_code=404, detail="Agent call not found")
    return call

@app.get("/api/v2/trace/enabled")
async def api_get_trace_enabled():
    return {"enabled": get_trace_enabled()}


@app.post("/api/v2/trace/enabled")
async def api_set_trace_enabled(body: Dict[str, Any]):
    enabled = bool(body.get("enabled", True))
    set_trace_enabled(enabled)
    return {"enabled": get_trace_enabled()}


@app.delete("/api/v2/traces")
async def api_clear_traces():
    clear_traces()
    return {"ok": True}


# ---------------------------------------------------------------------------
# Export endpoints
# ---------------------------------------------------------------------------

@app.get("/api/v2/trace/download")
async def api_download_trace():
    """Download the raw trace.db file."""
    if not DB_PATH.exists():
        raise HTTPException(status_code=404, detail="Trace database not found")
    return FileResponse(
        path=str(DB_PATH),
        media_type="application/octet-stream",
        filename="trace.db",
        headers={"Content-Disposition": "attachment; filename=\"trace.db\""},
    )


async def _export_and_zip(
    session_id: Optional[str] = None,
    raw_only: bool = False,
    include_subagents: bool = False,
    include_archived: bool = False,
    timeout: int = 120,
) -> io.BytesIO:
    """Run export_training.py into a temp dir and return an in-memory zip."""
    import shutil

    tmpdir = tempfile.mkdtemp(prefix="export_training_")
    try:
        cmd = [
            sys.executable, str(EXPORT_TRAINING_SCRIPT),
            "--db", str(DB_PATH),
            "--out", tmpdir,
        ]
        if session_id:
            cmd.extend(["--session", session_id])
        if raw_only:
            cmd.append("--raw-only")
        if include_subagents:
            cmd.append("--export-subagents")
        if include_archived:
            cmd.append("--include-archived")

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            raise HTTPException(
                status_code=500,
                detail=f"Export timed out after {timeout}s",
            )

        if proc.returncode != 0:
            err_text = stderr.decode("utf-8", errors="replace") if stderr else ""
            raise HTTPException(
                status_code=500,
                detail=f"Export failed (exit {proc.returncode}): {err_text[:1000]}",
            )

        output_files = sorted(Path(tmpdir).glob("*"))
        if not output_files:
            raise HTTPException(
                status_code=404,
                detail="No output generated (session may have no main-thread data)",
            )

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for fp in output_files:
                if fp.is_file():
                    zf.write(fp, fp.name)
        buf.seek(0)
        return buf
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


@app.get("/api/v2/sessions/{session_id}/export/training")
async def api_export_session_training(
    session_id: str,
    raw_only: bool = False,
):
    """Export a single session's training data as a zip file."""
    sess = get_session(session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found")

    zip_buf = await _export_and_zip(
        session_id=session_id,
        raw_only=raw_only,
        timeout=120,
    )
    short_id = session_id[:8]
    return StreamingResponse(
        iter([zip_buf.getvalue()]),
        media_type="application/zip",
        headers={
            "Content-Disposition": f"attachment; filename=\"session_{short_id}_export.zip\"",
        },
    )


@app.get("/api/v2/export/training")
async def api_export_all_training(
    raw_only: bool = False,
    include_subagents: bool = False,
    include_archived: bool = False,
):
    """Export all sessions' training data as a zip file."""
    zip_buf = await _export_and_zip(
        raw_only=raw_only,
        include_subagents=include_subagents,
        include_archived=include_archived,
        timeout=300,
    )
    return StreamingResponse(
        iter([zip_buf.getvalue()]),
        media_type="application/zip",
        headers={
            "Content-Disposition": "attachment; filename=\"training_export.zip\"",
        },
    )

# Define ANSI color codes for terminal output
class Colors:
    CYAN = "\033[96m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    MAGENTA = "\033[95m"
    RESET = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    DIM = "\033[2m"
def log_request_beautifully(method, path, claude_model, openai_model, num_messages, num_tools, status_code):
    """Log requests in a beautiful, twitter-friendly format showing Claude to OpenAI mapping."""
    # Format the Claude model name nicely
    claude_display = f"{Colors.CYAN}{claude_model}{Colors.RESET}"
    
    # Extract endpoint name
    endpoint = path
    if "?" in endpoint:
        endpoint = endpoint.split("?")[0]
    
    # Extract just the OpenAI model name without provider prefix
    openai_display = openai_model
    if "/" in openai_display:
        openai_display = openai_display.split("/")[-1]
    openai_display = f"{Colors.GREEN}{openai_display}{Colors.RESET}"
    
    # Format tools and messages
    tools_str = f"{Colors.MAGENTA}{num_tools} tools{Colors.RESET}"
    messages_str = f"{Colors.BLUE}{num_messages} messages{Colors.RESET}"
    
    # Format status code
    status_str = f"{Colors.GREEN}✓ {status_code} OK{Colors.RESET}" if status_code == 200 else f"{Colors.RED}✗ {status_code}{Colors.RESET}"
    

    # Put it all together in a clear, beautiful format
    log_line = f"{Colors.BOLD}{method} {endpoint}{Colors.RESET} {status_str}"
    model_line = f"{claude_display} → {openai_display} {tools_str} {messages_str}"
    
    # Print to console
    print(log_line)
    print(model_line)
    sys.stdout.flush()

if __name__ == "__main__":
    import sys
    if "--trace-disabled" in sys.argv:
        set_trace_enabled(False)
        logger.warning("Trace collection DISABLED by --trace-disabled flag")
    if "--help" in sys.argv:
        print("Usage: python server.py [--trace-disabled]")
        print("  --trace-disabled  Start with trajectory collection turned off")
        print()
        print("Or run with: uvicorn server:app --host 0.0.0.0 --port 8082 --workers 4")
        sys.exit(0)

    # Concurrency tuning via env:
    #   PORT          — listen port (default 8082)
    #   WORKERS       — number of worker processes (default 1; set >1 for true
    #                   multi-core parallelism, but each worker is a separate
    #                   process so module-level state is not shared)
    #   BACKLOG       — TCP accept queue size (default 2048)
    #   LIMIT_CONCURRENCY — max in-flight requests per worker before 503 (None=no limit)
    port = int(os.environ.get("PORT", "8082"))
    workers = int(os.environ.get("WORKERS", "16"))
    backlog = int(os.environ.get("BACKLOG", "2048"))
    limit_concurrency_env = os.environ.get("LIMIT_CONCURRENCY")
    limit_concurrency = int(limit_concurrency_env) if limit_concurrency_env else None

    uvicorn_kwargs = {
        "host": "0.0.0.0",
        "port": port,
        "log_level": "error",
        "backlog": backlog,
        "timeout_keep_alive": 75,
    }
    if limit_concurrency is not None:
        uvicorn_kwargs["limit_concurrency"] = limit_concurrency

    if workers > 1:
        # Multi-worker mode requires an import string, not the app object.
        uvicorn.run("server:app", workers=workers, **uvicorn_kwargs)
    else:
        uvicorn.run(app, **uvicorn_kwargs)
