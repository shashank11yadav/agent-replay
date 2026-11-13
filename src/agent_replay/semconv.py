"""Pinned OpenTelemetry GenAI semantic convention names (Development status).

This is the only module allowed to spell out ``gen_ai.*`` keys; everything else
imports the constants from here so a convention rename touches one file.
"""

from __future__ import annotations

GEN_AI_OPERATION_NAME = "gen_ai.operation.name"
GEN_AI_PROVIDER_NAME = "gen_ai.provider.name"
GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
GEN_AI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
GEN_AI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
GEN_AI_TOOL_NAME = "gen_ai.tool.name"
GEN_AI_TOOL_CALL_ID = "gen_ai.tool.call.id"
GEN_AI_TOOL_CALL_ARGUMENTS = "gen_ai.tool.call.arguments"
GEN_AI_TOOL_CALL_RESULT = "gen_ai.tool.call.result"
ERROR_TYPE = "error.type"

OP_INVOKE_AGENT = "invoke_agent"
OP_GENERATE_CONTENT = "generate_content"
OP_EXECUTE_TOOL = "execute_tool"
