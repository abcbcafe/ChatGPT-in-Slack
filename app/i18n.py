from typing import Optional

from slack_bolt import BoltContext

# All the supported languages for Slack app as of March 2023
_locale_to_lang = {
    "en-US": "English",
    "en-GB": "English",
    "de-DE": "German",
    "es-ES": "Spanish",
    "es-LA": "Spanish",
    "fr-FR": "French",
    "it-IT": "Italian",
    "pt-BR": "Portuguese",
    "ru-RU": "Russian",
    "ja-JP": "Japanese",
    "zh-CN": "Chinese",
    "zh-TW": "Chinese",
    "ko-KR": "Korean",
}


def from_locale_to_lang(locale: Optional[str]) -> Optional[str]:
    if locale is None:
        return None
    return _locale_to_lang.get(locale)


_translation_result_cache = {}


def translate(*, api_key: Optional[str], context: BoltContext, text: str) -> str:
    return text
