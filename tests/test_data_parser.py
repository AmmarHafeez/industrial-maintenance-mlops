from __future__ import annotations

from industrial_maintenance_mlops.data.parser import CMAPSS_COLUMNS, read_cmapss_file, read_rul_file


def test_parser_handles_tiny_cmapss_like_text(tiny_cmapss_file):
    frame = read_cmapss_file(tiny_cmapss_file)

    assert list(frame.columns) == CMAPSS_COLUMNS
    assert frame.shape == (6, 26)
    assert frame["unit_number"].tolist()[:3] == [1, 1, 1]
    assert frame["time_in_cycles"].tolist()[:3] == [1, 2, 3]


def test_rul_file_uses_unit_number_index(tmp_path):
    path = tmp_path / "RUL_FD001.txt"
    path.write_text("7\n11\n", encoding="utf-8")

    values = read_rul_file(path)

    assert values.index.tolist() == [1, 2]
    assert values.tolist() == [7, 11]
