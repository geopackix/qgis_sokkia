"""SDR33 (Sokkia format) export message assembly and protocol helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from .coordinate import Coordinate
from .header import Header
from .job import Job


def get_sdr33_checksum(data: str) -> str:
    """Calculate the SDR33 checksum."""
    total = 0
    for ch in data:
        code = ord(ch)
        if code not in (0x0D, 0x0A, 0x02, 0x03):
            total += code
    checksum = total % 65536
    return str(checksum).zfill(5)


def get_sdr33_message(rawdata: str) -> str:
    """Wrap raw record data into a valid SDR33 protocol message."""
    msg = chr(0x02) + chr(0x0A) + rawdata + chr(0x03)
    msg += get_sdr33_checksum(msg)
    return msg


class Sdr33Export:
    """Main class for creating SDR33 format messages for Sokkia total stations."""

    def __init__(self, job_name: str) -> None:
        self._job = Job(job_name)
        self._header = Header()
        self._coordinates: List[Coordinate] = []

    def add_coordinate(self, point: Coordinate) -> bool:
        self._coordinates.append(point)
        return True

    def get_message(self) -> str:
        """Return the complete SDR33 format message string."""
        raw = self._header.get_message() + "\n"
        raw += self._job.get_message() + "\n"
        for coord in self._coordinates:
            raw += coord.get_message() + "\n"
        return get_sdr33_message(raw)

    @staticmethod
    def from_geojson(path: str) -> Optional["Sdr33Export"]:
        """Create an Sdr33Export from a GeoJSON Point feature collection."""
        raw = Path(path).read_text(encoding="utf-8")
        geojson = json.loads(raw)

        name = geojson.get("name", "")
        export = Sdr33Export(name)

        index_n = 1
        index_e = 0
        index_z = 2

        for i, feature in enumerate(geojson.get("features", [])):
            props = feature.get("properties", {})
            coords = feature["geometry"]["coordinates"]
            point = Coordinate(
                point_name=props.get("name", str(i)),
                northing=coords[index_n],
                easting=coords[index_e],
                elevation=coords[index_z] if len(coords) > index_z else 0,
                description=props.get("description", ""),
            )
            export.add_coordinate(point)

        return export
