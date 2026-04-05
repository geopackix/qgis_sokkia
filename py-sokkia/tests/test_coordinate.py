"""Tests for sdr33.coordinate."""

from sdr33.coordinate import Coordinate


class TestCoordinateLengths:
    def test_short_northing_length(self):
        c = Coordinate("Testpunkt", 0.01, 0, 0, "Description")
        n_byte = c.get_northing_bytes()
        assert len(n_byte) == 16

    def test_long_northing_length(self):
        c = Coordinate("Testpunkt", 99.987654321987654321, 0, 0, "Description")
        n_byte = c.get_northing_bytes()
        assert len(n_byte) == 16

    def test_message_is_string(self):
        c = Coordinate("Testpunkt", 128.25668999666333555888, 0, 0, "Description")
        assert isinstance(c.get_message(), str)

    def test_message_length_with_overflow_values(self):
        c = Coordinate(
            "Testpunktwithtoolongpouintname",
            0.00000112423534564564576567,
            -18.345676354,
            9999999999999999999999999999999,
            "Description with too long desc data of type string",
        )
        assert len(c.get_message()) == 84
