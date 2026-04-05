"""SDR33 (Sokkia format) derivation codes indicating the type or source of records."""

from enum import Enum


class DerivationCode(str, Enum):
    BLANK = "  "
    KEYBOARD_INPUT = "KI"
    COORDINATES_PROGRAM = "CO"
    TOPOGRAPHY_PROGRAM = "TP"
    EXTERNALLY_DERIVED = "XD"
    UNKNOWN1 = "ED"  # not documented
    NOT_MEASURED = "NM"
