from __future__ import annotations

import numpy as np

from industrial_maintenance_mlops.monitoring.drift import (
    compare_window_to_reference,
    compute_reference_stats,
)


def test_drift_statistics_flags_large_mean_shift():
    reference_windows = np.array(
        [
            [[1.0, 2.0], [2.0, 3.0]],
            [[2.0, 3.0], [3.0, 4.0]],
        ]
    )
    stats = compute_reference_stats(reference_windows, ["sensor_1", "sensor_2"])

    result = compare_window_to_reference(
        np.array([[10.0, 2.0], [11.0, 3.0]]),
        stats,
        z_threshold=2.0,
    )

    assert result.is_drifted is True
    assert "sensor_1" in result.drifted_features
    assert result.max_abs_z_score > 2.0
