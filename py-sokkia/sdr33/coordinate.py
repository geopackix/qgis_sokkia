"""SDR33 (Sokkia format) Coordinate class for point coordinate objects."""

from sdr33.derivation_code import DerivationCode
from sdr33.sdr33_tools import fill_up


class Coordinate:
    """A single SDR33 coordinate record (type code ``08``).

    Fixed message length: 84 bytes.
    """

    TYPE_CODE = "08"
    DATA_LENGTH = 84

    def __init__(
        self,
        point_name: str,
        northing: float,
        easting: float,
        elevation: float,
        description: str = "",
    ) -> None:
        """Create a SDR33 coordinate object.

        Args:
            point_name: Point name (max 16 characters).
            northing: North value.
            easting: East value.
            elevation: Elevation value.
            description: Point description (max 16 characters).
        """
        self._derivation_code = DerivationCode.KEYBOARD_INPUT
        self._point_id = point_name
        self._northing = round(northing, 4)
        self._easting = round(easting, 4)
        self._elevation = round(elevation, 4)
        self._description = description

    def get_point_id_bytes(self) -> str:
        return fill_up(self._point_id, 16)

    def get_northing_bytes(self) -> str:
        return fill_up(str(self._northing), 16)

    def get_easting_bytes(self) -> str:
        return fill_up(str(self._easting), 16)

    def get_elevation_bytes(self) -> str:
        return fill_up(str(self._elevation), 16)

    def get_description_bytes(self) -> str:
        return fill_up(self._description, 16)

    def get_message(self) -> str:
        """Return the formatted SDR33 coordinate record string."""
        return (
            self.TYPE_CODE
            + self._derivation_code.value
            + self.get_point_id_bytes()
            + self.get_northing_bytes()
            + self.get_easting_bytes()
            + self.get_elevation_bytes()
            + self.get_description_bytes()
        )
