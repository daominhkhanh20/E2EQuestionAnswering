import unicodedata
import re
import string


def process_text(text: str):
    text = unicodedata.normalize('NFC', text).lower()
    return text


def preprocess_qa_text(text):
    text = re.sub("([,!:?\-+()])", r' \1 ', text)
    text = re.sub('\s{2,}', ' ', text)
    text = text.strip()
    return text


def preprocess_answer(context, answer, answer_start):
    if answer in context:
        idx = context.find(answer)
        while idx != -1 and idx < answer_start - 1:
            if answer_start is None:
                break
            elif answer_start is not None and idx < answer_start - 1:
                idx = context.find(answer, idx + 1)
        start_index = idx
        if start_index > 0 and context[start_index - 1] == " ":
            return answer.strip()
        while start_index > 0 and (context[start_index - 1].isalpha() or context[start_index - 1].isnumeric()):
            start_index -= 1
            answer = context[start_index] + answer

    return answer.strip()
