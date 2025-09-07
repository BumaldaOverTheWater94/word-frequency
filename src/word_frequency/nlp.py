import spacy
from loguru import logger
from spacy.language import Language
from spacy.tokenizer import Tokenizer
from spacy.tokens import Doc
from spacy.util import compile_infix_regex, compile_prefix_regex, compile_suffix_regex


@Language.component("custom_fallback_lemmatizer")
def custom_fallback_lemmatizer(doc: Doc) -> Doc:
    """
    This component checks for specific lemmatization errors and corrects them.
    It's a fallback for when the model's default lemmatizer makes a mistake.
    """
    for token in doc:
        is_er_or_ed_word = token.text.endswith("er") or token.text.endswith("ed")
        is_bad_lemma = token.lemma_.endswith("e")

        if is_er_or_ed_word and is_bad_lemma and token.lemma_[:-1] == token.text[:-2]:
            token.lemma_ = token.text

    return doc


def load_model(max_length: int) -> spacy.language.Language:
    # need parser for NER and need tagger for lemmatizer
    # even attribute_ruler is needed for lemmatization
    nlp = spacy.load("en_core_web_trf", exclude=[])
    nlp.max_length = max_length
    nlp.add_pipe("custom_fallback_lemmatizer", after="lemmatizer")

    # Enhanced custom tokenizer to fix problematic tokenization
    # Enhanced prefix patterns - split these at the beginning of tokens
    custom_prefixes = [
        r"[\.\,\;\:\!\?\[\]\(\)\{\}]",  # Punctuation at start
        r"[\"\'`]",  # Quotes at start
        r"\([0-9]+\)",  # Standalone parentheses with numbers at start
        r"[~\-]",  # Tilde and dash at start
        r"[\"=\+]",  # Quote-equals, plus at start
    ]

    # Enhanced suffix patterns - split these at the end of tokens
    custom_suffixes = [
        r"[\.\,\;\:\!\?\[\]\(\)\{\}]",  # Punctuation at end
        r"[\"\'`]",  # Quotes at end
        r"(?<=[a-zA-Z])\d+",  # Numbers after letters
        r"(?<=[a-zA-Z])\([0-9]+\)",  # Parenthetical numbers after letters
        r"(?<=[0-9])\%",  # Percent after numbers
        r"\([0-9]+\)",  # Standalone parentheses with numbers at end
        r"(?<=[a-zA-Z])\.",  # Periods after single letters
        r"(?<=[a-zA-Z])[-~,]",  # Hyphens, tildes, and commas after letters
        r"[\"\']+[^\"\']*[\"\']+",  # Complex quote patterns
        r"[\"=\+]",  # Quote-equals, plus at end
    ]

    # Enhanced infix patterns - split within tokens
    custom_infixes = [
        r"(?<=[a-zA-Z])\.(?=[a-zA-Z])",  # Periods between any letters (custom rule)
        r"(?<=[a-zA-Z])\?(?=[a-zA-Z])",  # Question marks between any letters (custom rule)
        r"(?<=[A-Za-z])\(",  # Opening paren after letters
        r"(?<=[0-9])\)",  # Closing paren after numbers
        r"(?<=[A-Za-z0-9])\(",  # Opening paren after alphanumeric
        r"(?<=[0-9])(?=[A-Za-z])",  # Letters after numbers
        r"(?<=[A-Za-z])(?=[0-9])",  # Numbers after letters
        r"(?<=[0-9])%\(",  # Percent-paren combo after numbers
        r"(?<=[a-zA-Z])[!~-](?=[a-zA-Z])",  # Exclamation, tildes, and hyphens within compound words
        r"\([0-9]+\)",  # Standalone parentheses with numbers within tokens
        r"[\"=\+]",  # Quote-equals, plus within tokens
    ]

    # Essential spaCy patterns that don't conflict
    essential_spacy_infixes = [
        r"(?<=[0-9])[+\-\*^](?=[0-9-])",  # Math operators between numbers
        r"(?<=[a-zA-Z]),(?=[a-zA-Z])",  # Commas between letters
        r"(?<=[a-zA-Z])(?:--|––|—)(?=[a-zA-Z])",  # Hyphens and dashes between letters
        r"(?<=[a-zA-Z0-9])[:<>=/](?=[a-zA-Z])",  # Symbols between letters/numbers
    ]

    # Combine patterns
    prefixes = custom_prefixes + list(nlp.Defaults.prefixes or [])
    suffixes = custom_suffixes + list(nlp.Defaults.suffixes or [])
    infixes = custom_infixes + essential_spacy_infixes

    # Compile regex patterns
    prefix_regex = compile_prefix_regex(prefixes)
    suffix_regex = compile_suffix_regex(suffixes)
    infix_regex = compile_infix_regex(infixes)

    # Create a brand new tokenizer with no built-in rules
    new_tokenizer = Tokenizer(
        nlp.vocab,
        rules={},  # Empty rules dict
        prefix_search=prefix_regex.search,
        suffix_search=suffix_regex.search,
        infix_finditer=infix_regex.finditer,
        token_match=None,  # No token matching
    )

    # Replace the model's tokenizer with our custom one
    nlp.tokenizer = new_tokenizer

    logger.info(f"Loaded model with max length {max_length}")
    return nlp
