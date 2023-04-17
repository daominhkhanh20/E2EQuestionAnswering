import unicodedata
import re


def process_text(text: str):
    text = unicodedata.normalize('NFC', text).lower()
    return text
