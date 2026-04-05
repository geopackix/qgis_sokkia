"""SDR33 (Sokkia format) utility functions."""


def fill_up(value: str, length: int) -> str:
    """Pad or truncate a string to an exact fixed width."""
    if len(value) > length:
        return value[:length]
    return value.ljust(length)
