import re

import spacy


def filter_token(token: spacy.tokens.Token) -> bool:
    """
    This function provides comprehensive filtering to prevent problematic tokens
    like "head," from appearing in the final word frequency output.

    Args:
        token: spaCy Token object to filter

    Returns:
        bool: True if token should be included, False if it should be filtered out
    """

    basic_filters = (
        token.is_ascii
        and token.ent_type_ == ""
        and not token.is_space
        and not token.is_punct
        and not token.is_digit
        and not token.is_quote
        and not token.is_bracket
        and not token.is_currency
        and not token.like_num
        and not token.like_url
        and not token.like_email
        and not token.is_stop
        and token.pos_ not in ["SYM", "PUNCT", "INTJ", "NUM"]
    )

    if not basic_filters:
        return False

    lemma = token.lemma_.strip()

    # Define punctuation and symbols
    symbols = [
        "=",
        "+",
        "-",
        "~",
        ".",
        ",",
        "!",
        "?",
        '"',
        "'",
        "(",
        ")",
        "[",
        "]",
        "{",
        "}",
        ":",
        ";",
    ]

    if lemma in symbols or not lemma:
        return False

    if len(lemma) <= 3 and any(c in lemma for c in symbols):
        return False

    if any(c in lemma for c in symbols):
        return False

    excluded_unicode = [
        "\u200b",  # Zero-width space
        "\u200c",  # Zero-width non-joiner
        "\u200d",  # Zero-width joiner
        "\u200e",  # Left-to-right mark
        "\u200f",  # Right-to-left mark
        "\ufeff",  # Zero-width no-break space (BOM)
        "\u00a0",  # Non-breaking space
    ]

    if any(char in lemma for char in excluded_unicode):
        return False

    # No English words repeat a letter more than twice (filters onomatopoeia)
    if re.search(r"(.)\1\1+", lemma):
        return False

    # Every English word has at least one vowel
    if not re.search(r"[aeiouy]", lemma):
        return False

    # no english word is longer than 46 characters
    if len(lemma) > 46:
        return False

    return True
