import logging
import os
import re
import time

from openai.error import Timeout
from slack_bolt import App, Ack, BoltContext, BoltResponse
from slack_bolt.request.payload_utils import is_event
from slack_sdk.http_retry import RateLimitErrorRetryHandler
from slack_sdk.web import WebClient

from app.LlmAdapter import LlmAdapter
from app.env import (
    LLM_TIMEOUT_SECONDS,
    SYSTEM_TEXT,
    TRANSLATE_MARKDOWN,
    SLACK_APP_LOG_LEVEL,
    USE_SLACK_LANGUAGE,
)
from app.openai_ops import (
    format_openai_message_content,
    build_system_text,
    messages_within_context_window,
)
from app.slack_ops import (
    find_parent_message,
    is_this_app_mentioned,
    post_wip_message,
    update_wip_message,
    extract_state_value,
)

TIMEOUT_ERROR_MESSAGE = (
    f":warning: Apologies! It seems that OpenAI didn't respond within the {LLM_TIMEOUT_SECONDS}-second timeframe. "
    "Please try your request again later. "
    "If you wish to extend the timeout limit, "
    "you may consider deploying this app with customized settings on your infrastructure. :bow:"
)
DEFAULT_LOADING_TEXT = ":hourglass_flowing_sand: Wait a second, please ..."


class SlackFrontend:
    def __init__(self, llm_adapter: LlmAdapter):
        self.llm_adapter = llm_adapter

    def respond_to_app_mention(
        self,
        context: BoltContext,
        payload: dict,
        client: WebClient,
        logger: logging.Logger,
    ):
        if payload.get("thread_ts") is not None:
            parent_message = find_parent_message(
                client, context.channel_id, payload.get("thread_ts")
            )
            if parent_message is not None and is_this_app_mentioned(
                context, parent_message
            ):
                # The message event handler will reply to this
                return

        wip_reply = None
        # Replace placeholder for Slack user ID in the system prompt
        system_text = build_system_text(SYSTEM_TEXT, TRANSLATE_MARKDOWN, context)
        messages = [{"role": "system", "content": system_text}]

        try:
            user_id = context.actor_user_id or context.user_id

            if payload.get("thread_ts") is not None:
                # Mentioning the bot user in a thread
                replies_in_thread = client.conversations_replies(
                    channel=context.channel_id,
                    ts=payload.get("thread_ts"),
                    include_all_metadata=True,
                    limit=1000,
                ).get("messages", [])
                for reply in replies_in_thread:
                    reply_text = reply.get("text")
                    messages.append(
                        {
                            "role": (
                                "assistant"
                                if "user" in reply
                                and reply["user"] == context.bot_user_id
                                else "user"
                            ),
                            "content": (
                                f"<@{reply['user'] if 'user' in reply else reply['username']}>: "
                                + format_openai_message_content(
                                    reply_text, TRANSLATE_MARKDOWN
                                )
                            ),
                        }
                    )
            else:
                # Strip bot Slack user ID from initial message
                msg_text = re.sub(f"<@{context.bot_user_id}>\\s*", "", payload["text"])
                messages.append(
                    {
                        "role": "user",
                        "content": f"<@{user_id}>: "
                        + format_openai_message_content(msg_text, TRANSLATE_MARKDOWN),
                    }
                )

            loading_text = DEFAULT_LOADING_TEXT

            wip_reply = post_wip_message(
                client=client,
                channel=context.channel_id,
                thread_ts=payload["ts"],
                loading_text=loading_text,
                messages=messages,
                user=context.user_id,
            )
            (
                messages,
                num_context_tokens,
                max_context_tokens,
            ) = messages_within_context_window(messages, context=context)

            output = self.llm_adapter.generate_text(messages)
            update_wip_message(
                client=client,
                channel=context.channel_id,
                ts=wip_reply["message"]["ts"],
                text=output,
                messages=messages,
                user=context.user_id,
            )

        except Timeout:
            if wip_reply is not None:
                text = (
                    (
                        wip_reply.get("message", {}).get("text", "")
                        if wip_reply is not None
                        else ""
                    )
                    + "\n\n"
                    + TIMEOUT_ERROR_MESSAGE
                )
                client.chat_update(
                    channel=context.channel_id,
                    ts=wip_reply["message"]["ts"],
                    text=text,
                )
        except Exception as e:
            text = (
                (
                    wip_reply.get("message", {}).get("text", "")
                    if wip_reply is not None
                    else ""
                )
                + "\n\n"
                + f":warning: Failed to start a conversation with Claudine: {e}"
            )
            logger.exception(text, e)
            if wip_reply is not None:
                client.chat_update(
                    channel=context.channel_id,
                    ts=wip_reply["message"]["ts"],
                    text=text,
                )

    def find_first_url_in_message(self, message):
        if "http://" in message:
            url = re.findall("http://.*", message)[0]
            return url
        else:
            return None

    def respond_to_new_message(
        self,
        context: BoltContext,
        payload: dict,
        client: WebClient,
        logger: logging.Logger,
    ):
        if (
            payload.get("bot_id") is not None
            and payload.get("bot_id") != context.bot_id
        ):
            # Skip a new message by a different app
            return

        wip_reply = None
        try:
            is_in_dm_with_bot = payload.get("channel_type") == "im"
            is_thread_for_this_app = False
            thread_ts = payload.get("thread_ts")
            if is_in_dm_with_bot is False and thread_ts is None:
                return

            messages_in_context = []
            if is_in_dm_with_bot is True and thread_ts is None:
                # In the DM with the bot; this is not within a thread
                past_messages = client.conversations_history(
                    channel=context.channel_id,
                    include_all_metadata=True,
                    limit=100,
                ).get("messages", [])
                past_messages.reverse()
                # Remove old messages
                for message in past_messages:
                    seconds = time.time() - float(message.get("ts"))
                    if seconds < 86400:  # less than 1 day
                        messages_in_context.append(message)
                is_thread_for_this_app = True
            else:
                # Within a thread
                messages_in_context = client.conversations_replies(
                    channel=context.channel_id,
                    ts=thread_ts,
                    include_all_metadata=True,
                    limit=1000,
                ).get("messages", [])
                if is_in_dm_with_bot is True:
                    # In the DM with this bot
                    is_thread_for_this_app = True
                else:
                    # In a channel
                    the_parent_message_found = False
                    for message in messages_in_context:
                        if message.get("ts") == thread_ts:
                            the_parent_message_found = True
                            is_thread_for_this_app = is_this_app_mentioned(
                                context, message
                            )
                            break
                    if the_parent_message_found is False:
                        parent_message = find_parent_message(
                            client, context.channel_id, thread_ts
                        )
                        if parent_message is not None:
                            is_thread_for_this_app = is_this_app_mentioned(
                                context, parent_message
                            )

            if is_thread_for_this_app is False:
                return

            messages = []
            user_id = context.actor_user_id or context.user_id
            last_assistant_idx = -1
            indices_to_remove = []
            for idx, reply in enumerate(messages_in_context):
                maybe_event_type = reply.get("metadata", {}).get("event_type")
                if maybe_event_type == "chat-gpt-convo":
                    if context.bot_id != reply.get("bot_id"):
                        # Remove messages by a different app
                        indices_to_remove.append(idx)
                        continue
                    maybe_new_messages = (
                        reply.get("metadata", {})
                        .get("event_payload", {})
                        .get("messages")
                    )
                    if maybe_new_messages is not None:
                        if len(messages) == 0 or user_id is None:
                            new_user_id = (
                                reply.get("metadata", {})
                                .get("event_payload", {})
                                .get("user")
                            )
                            if new_user_id is not None:
                                user_id = new_user_id
                        messages = maybe_new_messages
                        last_assistant_idx = idx

            if is_in_dm_with_bot is True or last_assistant_idx == -1:
                # To know whether this app needs to start a new convo
                if not next(
                    filter(lambda msg: msg["role"] == "system", messages), None
                ):
                    # Replace placeholder for Slack user ID in the system prompt
                    system_text = build_system_text(
                        SYSTEM_TEXT, TRANSLATE_MARKDOWN, context
                    )
                    messages.insert(0, {"role": "system", "content": system_text})

            filtered_messages_in_context = []
            for idx, reply in enumerate(messages_in_context):
                # Strip bot Slack user ID from initial message
                if idx == 0:
                    reply["text"] = re.sub(
                        f"<@{context.bot_user_id}>\\s*", "", reply["text"]
                    )
                if idx not in indices_to_remove:
                    filtered_messages_in_context.append(reply)
            if len(filtered_messages_in_context) == 0:
                return

            for reply in filtered_messages_in_context:
                msg_user_id = reply.get("user")
                reply_text = reply.get("text")
                messages.append(
                    {
                        "content": f"<@{msg_user_id}>: "
                        + format_openai_message_content(reply_text, TRANSLATE_MARKDOWN),
                        "role": (
                            "assistant"
                            if "user" in reply and reply["user"] == context.bot_user_id
                            else "user"
                        ),
                    }
                )

            loading_text = DEFAULT_LOADING_TEXT
            wip_reply = post_wip_message(
                client=client,
                channel=context.channel_id,
                thread_ts=(
                    payload.get("thread_ts") if is_in_dm_with_bot else payload["ts"]
                ),
                loading_text=loading_text,
                messages=messages,
                user=user_id,
            )

            (
                messages,
                num_context_tokens,
                max_context_tokens,
            ) = messages_within_context_window(messages, context=context)
            num_messages = len([msg for msg in messages if msg.get("role") != "system"])
            if num_messages == 0:
                update_wip_message(
                    client=client,
                    channel=context.channel_id,
                    ts=wip_reply["message"]["ts"],
                    text=f":warning: The previous message is too long ({num_context_tokens}/{max_context_tokens} prompt tokens).",
                    messages=messages,
                    user=context.user_id,
                )
            else:
                if self.find_first_url_in_message(messages) is not None:
                    url = self.find_first_url_in_message(messages)
                    anthropic_call = self.llm_adapter.summarize_webpage(url)
                else:
                    anthropic_call = self.llm_adapter.generate_text(messages)
                print(anthropic_call)

                # Update the WIP message with the response
                update_wip_message(
                    client=client,
                    channel=context.channel_id,
                    ts=wip_reply["message"]["ts"],
                    text=anthropic_call,
                    messages=messages,
                    user=user_id,
                )

        except Timeout:
            if wip_reply is not None:
                text = (
                    (
                        wip_reply.get("message", {}).get("text", "")
                        if wip_reply is not None
                        else ""
                    )
                    + "\n\n"
                    + TIMEOUT_ERROR_MESSAGE
                )
                client.chat_update(
                    channel=context.channel_id,
                    ts=wip_reply["message"]["ts"],
                    text=text,
                )
        except Exception as e:
            text = (
                (
                    wip_reply.get("message", {}).get("text", "")
                    if wip_reply is not None
                    else ""
                )
                + "\n\n"
                + f":warning: Failed to reply: {e}"
            )
            logger.exception(text, e)
            if wip_reply is not None:
                client.chat_update(
                    channel=context.channel_id,
                    ts=wip_reply["message"]["ts"],
                    text=text,
                )

    def start_chat_from_scratch(client: WebClient, body: dict):
        client.views_open(
            trigger_id=body.get("trigger_id"),
            view={
                "type": "modal",
                "callback_id": "chat-from-scratch",
                "title": {"type": "plain_text", "text": "ChatGPT"},
                "submit": {"type": "plain_text", "text": "Submit"},
                "close": {"type": "plain_text", "text": "Close"},
                "blocks": [
                    {
                        "type": "input",
                        "block_id": "prompt",
                        "label": {"type": "plain_text", "text": "Prompt"},
                        "element": {
                            "type": "plain_text_input",
                            "action_id": "input",
                            "multiline": True,
                        },
                    },
                ],
            },
        )

    def ack_chat_from_scratch_modal_submission(
        ack: Ack,
        payload: dict,
    ):
        prompt = extract_state_value(payload, "prompt").get("value")
        text = "\n".join(map(lambda s: f">{s}", prompt.split("\n")))
        ack(
            response_action="update",
            view={
                "type": "modal",
                "callback_id": "chat-from-scratch",
                "title": {"type": "plain_text", "text": "ChatGPT"},
                "close": {"type": "plain_text", "text": "Close"},
                "blocks": [
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f"{text}\n\nWorking on this now ... :hourglass:",
                        },
                    },
                ],
            },
        )

    def display_chat_from_scratch_result(
        self,
        client: WebClient,
        context: BoltContext,
        logger: logging.Logger,
        payload: dict,
    ):
        try:
            prompt = extract_state_value(payload, "prompt").get("value")
            text = "\n".join(map(lambda s: f">{s}", prompt.split("\n")))
            result = self.llm_adapter.generate_text(text)
            client.views_update(
                view_id=payload["id"],
                view={
                    "type": "modal",
                    "callback_id": "chat-from-scratch",
                    "title": {"type": "plain_text", "text": "Claudine"},
                    "close": {"type": "plain_text", "text": "Close"},
                    "blocks": [
                        {
                            "type": "section",
                            "text": {
                                "type": "mrkdwn",
                                "text": f"{text}\n\n{result}",
                            },
                        },
                    ],
                },
            )
        except Timeout:
            client.views_update(
                view_id=payload["id"],
                view={
                    "type": "modal",
                    "callback_id": "chat-from-scratch",
                    "title": {"type": "plain_text", "text": "Claudine"},
                    "close": {"type": "plain_text", "text": "Close"},
                    "blocks": [
                        {
                            "type": "section",
                            "text": {
                                "type": "mrkdwn",
                                "text": f"{text}\n\n{TIMEOUT_ERROR_MESSAGE}",
                            },
                        },
                    ],
                },
            )
        except Exception as e:
            logger.error(f"Failed to share a thread summary: {e}")
            client.views_update(
                view_id=payload["id"],
                view={
                    "type": "modal",
                    "callback_id": "chat-from-scratch",
                    "title": {"type": "plain_text", "text": "ChatGPT"},
                    "close": {"type": "plain_text", "text": "Close"},
                    "blocks": [
                        {
                            "type": "section",
                            "text": {
                                "type": "mrkdwn",
                                "text": f"{text}\n\n:warning: My apologies! "
                                f"An error occurred while generating the summary of this thread: {e}",
                            },
                        },
                    ],
                },
            )

    def register_listeners(self, app: App):
        # Chat with the bot
        app.event("app_mention")(ack=lambda: None, lazy=[self.respond_to_app_mention])
        app.event("message")(ack=lambda: None, lazy=[self.respond_to_new_message])
        # Free format chat
        app.action("templates-from-scratch")(
            ack=lambda: None,
            lazy=[self.start_chat_from_scratch],
        )
        app.view("chat-from-scratch")(
            ack=self.ack_chat_from_scratch_modal_submission,
            lazy=[self.display_chat_from_scratch_result],
        )

    # To reduce unnecessary workload in this app,
    # this before_authorize function skips message changed/deleted events.
    # Especially, "message_changed" events can be triggered many times when the app rapidly updates its reply.
    def before_authorize(
        self,
        body: dict,
        payload: dict,
        logger: logging.Logger,
        next_,
    ):
        if (
            is_event(body)
            and payload.get("type") == "message"
            and payload.get("subtype") in ["message_changed", "message_deleted"]
        ):
            logger.debug(
                "Skipped the following middleware and listeners "
                f"for this message event (subtype: {payload.get('subtype')})"
            )
            return BoltResponse(status=200, body="")
        next_()

    def start(self):
        from slack_bolt.adapter.socket_mode import SocketModeHandler

        logging.basicConfig(level=SLACK_APP_LOG_LEVEL)

        app = App(
            token=os.environ["SLACK_BOT_TOKEN"],
            before_authorize=self.before_authorize,
            process_before_response=True,
        )
        app.client.retry_handlers.append(RateLimitErrorRetryHandler(max_retry_count=2))

        self.register_listeners(app)

        if USE_SLACK_LANGUAGE is True:

            @app.middleware
            def set_locale(
                context: BoltContext,
                client: WebClient,
                next_,
            ):
                user_id = context.actor_user_id or context.user_id
                user_info = client.users_info(user=user_id, include_locale=True)
                context["locale"] = user_info.get("user", {}).get("locale")
                next_()

        handler = SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"])
        handler.start()
