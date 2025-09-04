import spacy
from loguru import logger
from spacy.util import compile_infix_regex, compile_prefix_regex, compile_suffix_regex


def load_model(max_length: int) -> spacy.language.Language:
    # need parser for NER and need tagger for lemmatizer
    # even attribute_ruler is needed for lemmatization
    nlp = spacy.load("en_core_web_trf", exclude=[])
    nlp.max_length = max_length

    # Enhanced custom tokenizer to fix problematic tokenization
    # Enhanced prefix patterns - split these at the beginning of tokens
    custom_prefixes = [
        r"[\.\,\;\:\!\?\[\]\(\)\{\}]",  # Punctuation at start
        r"[\"\'\`]",  # Quotes at start
        r"\([0-9]+\)",  # Standalone parentheses with numbers at start
        r"[~\-]",  # Tilde and dash at start
        r"[\"\=\+]",  # Quote-equals, plus at start
    ]

    # Enhanced suffix patterns - split these at the end of tokens
    custom_suffixes = [
        r"[\.\,\;\:\!\?\[\]\(\)\{\}]",  # Punctuation at end
        r"[\"\'\`]",  # Quotes at end
        r"(?<=[a-zA-Z])\d+",  # Numbers after letters
        r"(?<=[a-zA-Z])\([0-9]+\)",  # Parenthetical numbers after letters
        r"(?<=[0-9])\%",  # Percent after numbers
        r"\([0-9]+\)",  # Standalone parentheses with numbers at end
        r"(?<=[a-zA-Z])\.",  # Periods after single letters
        r"(?<=[a-zA-Z])[-~,]",  # Hyphens, tildes, and commas after letters
        r"[\"\'][^\"\']*[\"\']",  # Complex quote patterns
        r"[\"\=\+]",  # Quote-equals, plus at end
    ]

    # Enhanced infix patterns - split within tokens
    custom_infixes = [
        r"(?<=[A-Za-z])\(",  # Opening paren after letters
        r"(?<=[0-9])\)",  # Closing paren after numbers
        r"(?<=[A-Za-z0-9])\(",  # Opening paren after alphanumeric
        r"(?<=[0-9])(?=[A-Za-z])",  # Letters after numbers
        r"(?<=[A-Za-z])(?=[0-9])",  # Numbers after letters
        r"(?<=[0-9])%\(",  # Percent-paren combo after numbers
        r"(?<=[a-zA-Z])[!~-](?=[a-zA-Z])",  # Exclamation, tildes, and hyphens within compound words
        r"\([0-9]+\)",  # Standalone parentheses with numbers within tokens
        r"[\"\=\+]",  # Quote-equals, plus within tokens
    ]

    # Combine with existing spaCy defaults
    prefixes = list(nlp.Defaults.prefixes or []) + custom_prefixes
    suffixes = list(nlp.Defaults.suffixes or []) + custom_suffixes
    infixes = list(nlp.Defaults.infixes or []) + custom_infixes

    # Compile regex patterns
    prefix_regex = compile_prefix_regex(prefixes)
    suffix_regex = compile_suffix_regex(suffixes)
    infix_regex = compile_infix_regex(infixes)

    # Update tokenizer
    tokenizer: object = nlp.tokenizer
    tokenizer.prefix_search = prefix_regex.search  # type: ignore[attr-defined]
    tokenizer.suffix_search = suffix_regex.search  # type: ignore[attr-defined]
    tokenizer.infix_finditer = infix_regex.finditer  # type: ignore[attr-defined]

    logger.info(f"Loaded model with enhanced custom tokenizer, max length {max_length}")
    return nlp
