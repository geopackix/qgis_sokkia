"""Tests for sdr33.sdr33message."""

import json
import tempfile
from pathlib import Path

from sdr33 import Sdr33Export, Coordinate


class TestSdr33Export:
    def test_add_coordinate(self):
        export = Sdr33Export("TestJob")
        result = export.add_coordinate(Coordinate("P1", 100.0, 200.0, 50.0, "test"))
        assert result is True

    def test_get_message_returns_string(self):
        export = Sdr33Export("TestJob")
        export.add_coordinate(Coordinate("P1", 100.0, 200.0, 50.0, "test"))
        msg = export.get_message()
        assert isinstance(msg, str)
        # Message starts with STX (0x02) + LF (0x0A)
        assert msg[0] == chr(0x02)
        assert msg[1] == chr(0x0A)

    def test_from_geojson(self):
        geojson = {
            "name": "TestGeoJob",
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [10.5, 48.3, 500.0]},
                    "properties": {"name": "Station1", "description": "Ref point"},
                },
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [10.6, 48.4]},
                    "properties": {"name": "Station2"},
                },
            ],
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".geojson", delete=False, encoding="utf-8"
        ) as f:
            json.dump(geojson, f)
            f.flush()
            path = f.name

        export = Sdr33Export.from_geojson(path)
        assert export is not None
        msg = export.get_message()
        assert isinstance(msg, str)
        assert len(msg) > 0

        Path(path).unlink()

    def test_has_static_from_geojson(self):
        assert hasattr(Sdr33Export, "from_geojson")

    def test_has_add_coordinate(self):
        export = Sdr33Export("X")
        assert hasattr(export, "add_coordinate")

    def test_has_get_message(self):
        export = Sdr33Export("X")
        assert hasattr(export, "get_message")
