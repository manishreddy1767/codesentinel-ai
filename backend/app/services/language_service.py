LANGUAGE_ALIASES = {
    "c": "c",
    "c99": "c",
    "c11": "c",

    "cpp": "cpp",
    "c++": "cpp",
    "cxx": "cpp",
    "cc": "cpp",
}


SUPPORTED_LANGUAGES = {"c", "cpp"}


def normalize_language(language: str) -> str:
    normalized = language.strip().lower()

    if normalized in LANGUAGE_ALIASES:
        normalized = LANGUAGE_ALIASES[normalized]

    if normalized not in SUPPORTED_LANGUAGES:
        raise ValueError(
            f"Unsupported language: {language}. "
            "Currently supported languages are C and C++."
        )

    return normalized