"""Tests for sdr33.header."""

from sdr33.header import Header


class TestHeader:
    def test_message_length(self):
        h = Header()
        assert len(h.get_message()) == 46
