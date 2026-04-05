"""SDR33 (Sokkia format) Job identifier class."""

from sdr33.derivation_code import DerivationCode
from sdr33.sdr33_tools import fill_up


class Job:
    """SDR33 job identifier record (type code ``10``).

    Fixed message length: 26 bytes.
    """

    TYPE_CODE = "10"
    DATA_LENGTH = 26

    def __init__(self, job_name: str = "") -> None:
        """Create a new SDR33 job object.

        Args:
            job_name: Name of the SDR33 job.
        """
        self._derivation_code = DerivationCode.NOT_MEASURED
        self._job_name = job_name
        self._settings = {
            "point_id_type": 1,
            "include_ele": 2,
            "atmos_correction": 1,
            "cr_correction": 1,
            "refraction_constant": 1,
            "sea_level_correction": 1,
        }

    def _get_job_name(self) -> str:
        return fill_up(self._job_name, 16)

    def _get_settings(self) -> str:
        s = self._settings
        raw = (
            str(s["point_id_type"])
            + str(s["include_ele"])
            + str(s["atmos_correction"])
            + str(s["cr_correction"])
            + str(s["refraction_constant"])
            + str(s["sea_level_correction"])
        )
        return fill_up(raw, 6)

    def get_message(self) -> str:
        """Return the formatted SDR33 job identifier record string."""
        return (
            self.TYPE_CODE
            + self._derivation_code.value
            + self._get_job_name()
            + self._get_settings()
        )
