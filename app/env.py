import os

DEFAULT_SYSTEM_TEXT = """
You are a bot in a slack chat room. You might receive messages from multiple people.
Format bold text *like this*, italic text _like this_ and strikethrough text ~like this~.
Slack user IDs match the regex `<@U.*?>`.
Your Slack user ID is <@{bot_user_id}>.
Each message has the author's Slack user ID prepended, like the regex `^<@U.*?>: ` followed by the message text.
"""
SYSTEM_TEXT = os.environ.get("OPENAI_SYSTEM_TEXT", DEFAULT_SYSTEM_TEXT)

DEFAULT_LLM_TIMEOUT_SECONDS = 30
LLM_TIMEOUT_SECONDS = int(
    os.environ.get("LLM_TIMEOUT_SECONDS", DEFAULT_LLM_TIMEOUT_SECONDS)
)

DEFAULT_LLM_MODEL = "claude-3-opus-20240229"
LLM_MODEL = os.environ.get("LLM_MODEL", DEFAULT_LLM_MODEL)

DEFAULT_LLM_TEMPERATURE = 1
LLM_TEMPERATURE = float(
    os.environ.get("LLM_TEMPERATURE", DEFAULT_LLM_TEMPERATURE)
)

DEFAULT_LLM_API_BASE = "https://api.anthropic.com/v1"
LLM_API_BASE = os.environ.get("LLM_API_BASE", DEFAULT_LLM_API_BASE)

USE_SLACK_LANGUAGE = os.environ.get("USE_SLACK_LANGUAGE", "true") == "true"

SLACK_APP_LOG_LEVEL = os.environ.get("SLACK_APP_LOG_LEVEL", "DEBUG")

TRANSLATE_MARKDOWN = os.environ.get("TRANSLATE_MARKDOWN", "false") == "true"