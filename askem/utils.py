import secrets
import string
import textwrap


def generate_api_key(length=32) -> str:
    characters = string.ascii_letters + string.digits
    api_key = "".join(secrets.choice(characters) for _ in range(length))
    return api_key


def wrap_print(text, width=150) -> None:
    print(textwrap.fill(text, width=width))
