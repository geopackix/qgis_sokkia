"""SDR33 (Sokkia format) Header class."""

from datetime import datetime, timezone

from sdr33.derivation_code import DerivationCode
from sdr33.sdr33_tools import fill_up


class Header:
    """SDR33 protocol header record (type code ``00``).

    Fixed message length: 46 bytes.
    """

    TYPE_CODE = "00"
    DATA_LENGTH = 46

    def __init__(self) -> None:
        self._derivation_code = DerivationCode.NOT_MEASURED
        self._version = "SDR33 V04-04.02"
        self._serial_number = ""
        self._date_time = datetime.now(timezone.utc).isoformat()
        self._settings = {
            "angle_unit": 2,
            "distance_unit": 1,
            "pressure_unit": 3,
            "temp_unit": 1,
            "coord_prompt_option": 1,
            "angles_left_right_option": 1,
        }

    def _get_version_bytes(self) -> str:
        return fill_up(self._version, 16)

    def _get_serial_number_bytes(self) -> str:
        return fill_up(self._serial_number, 4)

    def _get_date_time_bytes(self) -> str:
        return fill_up(self._date_time, 16)

    def _get_settings(self) -> str:
        s = self._settings
        raw = (
            str(s["angle_unit"])
            + str(s["distance_unit"])
            + str(s["pressure_unit"])
            + str(s["temp_unit"])
            + str(s["coord_prompt_option"])
            + str(s["angles_left_right_option"])
        )
        return fill_up(raw, 6)

    def get_message(self) -> str:
        """Return the formatted SDR33 header record string."""
        return (
            self.TYPE_CODE
            + self._derivation_code.value
            + self._get_version_bytes()
            + self._get_serial_number_bytes()
            + self._get_date_time_bytes()
            + self._get_settings()
        )
