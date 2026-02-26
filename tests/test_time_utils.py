"""Tests for chester.time_utils."""
import pytest

from chester.time_utils import estimate_slurm_time, format_slurm_time, parse_slurm_time


class TestParseSlurmTime:
    def test_hms(self):
        assert parse_slurm_time("2:00:00") == 7200

    def test_hm(self):
        assert parse_slurm_time("2:30") == 9000

    def test_hours_only(self):
        assert parse_slurm_time("3") == 10800

    def test_zero(self):
        assert parse_slurm_time("0") == 0

    def test_days_hms(self):
        assert parse_slurm_time("1-12:30:00") == 86400 + 12 * 3600 + 30 * 60

    def test_days_hm(self):
        assert parse_slurm_time("2-06:00") == 2 * 86400 + 6 * 3600

    def test_days_hours(self):
        assert parse_slurm_time("1-0") == 86400

    def test_72_hours(self):
        assert parse_slurm_time("72:00:00") == 72 * 3600


class TestFormatSlurmTime:
    def test_simple_hours(self):
        assert format_slurm_time(7200) == "02:00:00"

    def test_with_days(self):
        assert format_slurm_time(86400 + 3600) == "1-01:00:00"

    def test_zero(self):
        assert format_slurm_time(0) == "00:00:00"

    def test_rounds_up(self):
        assert format_slurm_time(3600.1) == "01:00:01"

    def test_72_hours(self):
        assert format_slurm_time(72 * 3600) == "3-00:00:00"


class TestEstimateSlurmTime:
    def test_linear_double(self):
        result = estimate_slurm_time(x=100, y="0-2:00:00", x_target=200, b="0")
        assert parse_slurm_time(result) == 4 * 3600

    def test_linear_half(self):
        result = estimate_slurm_time(x=100, y="2:00:00", x_target=50)
        assert parse_slurm_time(result) == 3600

    def test_with_offset(self):
        # 100 epochs took 2:10:00, of which 0:10:00 is startup
        # slope = (7800 - 600) / 100 = 72 sec/epoch
        # target = 72 * 200 + 600 = 15000 sec = 4:10:00
        result = estimate_slurm_time(x=100, y="2:10:00", x_target=200, b="0:10:00")
        assert parse_slurm_time(result) == 15000

    def test_80_epochs(self):
        # 2hrs per 100 epochs -> 80 epochs = 1.6 hrs = 1:36:00
        result = estimate_slurm_time(x=100, y="2:00:00", x_target=80)
        assert parse_slurm_time(result) == 5760  # 1h 36m

    def test_zero_x_raises(self):
        with pytest.raises(ValueError, match="x must be positive"):
            estimate_slurm_time(x=0, y="1:00:00", x_target=100)

    def test_roundtrip_format(self):
        result = estimate_slurm_time(x=100, y="0-2:00:00", x_target=200)
        assert result == "04:00:00"
