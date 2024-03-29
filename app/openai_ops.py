import re
from typing import List, Dict, Tuple, Optional, Union

import anthropic
import tiktoken
from slack_bolt import BoltContext

from app.markdown import slack_to_markdown, markdown_to_slack

# ----------------------------
# Internal functions
# ----------------------------

_prompt_tokens_used_by_function_call_cache: Optional[int] = None


# Format message from Slack to send to OpenAI
def format_openai_message_content(content: str, translate_markdown: bool) -> str:
    if content is None:
        return None

    # Unescape &, < and >, since Slack replaces these with their HTML equivalents
    # See also: https://api.slack.com/reference/surfaces/formatting#escaping
    content = content.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")

    # Convert from Slack mrkdwn to markdown format
    if translate_markdown:
        content = slack_to_markdown(content)

    return content


def messages_within_context_window(
        messages: List[Dict[str, Union[str, Dict[str, str]]]],
        context: BoltContext,
) -> Tuple[List[Dict[str, Union[str, Dict[str, str]]]], int, int]:
    # Remove old messages to make sure we have room for max_tokens
    # See also: https://platform.openai.com/docs/guides/chat/introduction
    # > total tokens must be below the model’s maximum limit (e.g., 4096 tokens for gpt-3.5-turbo-0301)
    max_context_tokens = context_length() - 1
    num_context_tokens = 0  # Number of tokens in the context window just before the earliest message is deleted
    while (num_tokens := calculate_num_tokens(messages)) > max_context_tokens:
        removed = False
        for i, message in enumerate(messages):
            if message["role"] in ("user", "assistant", "function"):
                num_context_tokens = num_tokens
                del messages[i]
                removed = True
                break
        if not removed:
            # Fall through and let the OpenAI error handler deal with it
            break
    else:
        num_context_tokens = num_tokens

    return messages, num_context_tokens, max_context_tokens


def context_length(
) -> int:
    return 128000


# Adapted from https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
def calculate_num_tokens(
        messages: List[Dict[str, Union[str, Dict[str, str]]]]
) -> int:
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo-0301")
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    tokens_per_message = 3
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            if key == "function_call":
                num_tokens += 1
                num_tokens += len(encoding.encode(value["name"]))
                num_tokens += len(encoding.encode(value["arguments"]))
            else:
                num_tokens += len(encoding.encode(value))
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


# Format message from OpenAI to display in Slack
def format_assistant_reply(content: str, translate_markdown: bool) -> str:
    for o, n in [
        # Remove leading newlines
        ("^\n+", ""),
        # Remove prepended Slack user ID
        ("^<@U.*?>\\s?:\\s?", ""),
        # Remove OpenAI syntax tags since Slack doesn't render them in a message
        ("```\\s*[Rr]ust\n", "```\n"),
        ("```\\s*[Rr]uby\n", "```\n"),
        ("```\\s*[Ss]cala\n", "```\n"),
        ("```\\s*[Kk]otlin\n", "```\n"),
        ("```\\s*[Jj]ava\n", "```\n"),
        ("```\\s*[Gg]o\n", "```\n"),
        ("```\\s*[Ss]wift\n", "```\n"),
        ("```\\s*[Oo]objective[Cc]\n", "```\n"),
        ("```\\s*[Cc]\n", "```\n"),
        ("```\\s*[Cc][+][+]\n", "```\n"),
        ("```\\s*[Cc][Pp][Pp]\n", "```\n"),
        ("```\\s*[Cc]sharp\n", "```\n"),
        ("```\\s*[Mm][Aa][Tt][Ll][Aa][Bb]\n", "```\n"),
        ("```\\s*[Jj][Ss][Oo][Nn]\n", "```\n"),
        ("```\\s*[Ll]a[Tt]e[Xx]\n", "```\n"),
        ("```\\s*[Ll][Uu][Aa]\n", "```\n"),
        ("```\\s*[Cc][Mm][Aa][Kk][Ee]\n", "```\n"),
        ("```\\s*bash\n", "```\n"),
        ("```\\s*zsh\n", "```\n"),
        ("```\\s*sh\n", "```\n"),
        ("```\\s*[Ss][Qq][Ll]\n", "```\n"),
        ("```\\s*[Pp][Hh][Pp]\n", "```\n"),
        ("```\\s*[Pp][Ee][Rr][Ll]\n", "```\n"),
        ("```\\s*[Jj]ava[Ss]cript\n", "```\n"),
        ("```\\s*[Ty]ype[Ss]cript\n", "```\n"),
        ("```\\s*[Pp]ython\n", "```\n"),
    ]:
        content = re.sub(o, n, content)

    # Convert from OpenAI markdown to Slack mrkdwn format
    if translate_markdown:
        content = markdown_to_slack(content)

    return content


def build_system_text(
        system_text_template: str, translate_markdown: bool, context: BoltContext
):
    system_text = system_text_template.format(bot_user_id=context.bot_user_id)
    # Translate format hint in system prompt
    if translate_markdown is True:
        system_text = slack_to_markdown(system_text)
    return system_text
