"""Tests for sdr33.job."""

from sdr33.job import Job


class TestJob:
    def test_message_length(self):
        j = Job("MyTestJobWithLongName")
        assert len(j.get_message()) == 26
