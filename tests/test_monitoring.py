from pathlib import Path
import pandas as pd
import pytest

from src.monitoring import (
    generate_drifted_production_data,
    run_drift_monitoring
)
from src.retrain import is_retraining_triggered, execute_retraining


def test_generate_drifted_production_data():
    df = pd.DataFrame({
        "customer_id": ["C1", "C2", "C3", "C4"],
        "feature1": [1.0, 2.0, 3.0, 4.0],
        "feature2": [5.0, 6.0, 7.0, 8.0],
        "category_feature": ["A", "A", "B", "B"],
        "target": [0, 0, 1, 1]
    })
    drifted = generate_drifted_production_data(df, drift_factor=2.0, sample_size=10)
    assert len(drifted) == 10
    assert "feature1" in drifted.columns
    assert "feature2" in drifted.columns
    assert "category_feature" in drifted.columns


def test_run_drift_monitoring(tmp_path: Path):
    """
    Requirement 11:
    Verify reports/data_drift_report.html is created, is valid HTML, and is not empty.
    """
    ref_path = tmp_path / "ref.csv"
    report_path = tmp_path / "data_drift_report.html"
    flag_path = tmp_path / "drift_detected.flag"

    df = pd.DataFrame({
        "customer_id": [f"C{i}" for i in range(20)],
        "feature1": [float(i) for i in range(20)],
        "feature2": [float(i * 2) for i in range(20)],
        "category_feature": ["A"] * 10 + ["B"] * 10,
        "target": [i % 2 for i in range(20)]
    })
    df.to_csv(ref_path, index=False)

    res = run_drift_monitoring(
        reference_path=ref_path,
        output_report_path=report_path,
        flag_path=flag_path,
        set_drift_flag=True
    )

    assert report_path.exists()
    assert report_path.stat().st_size > 0
    html_text = report_path.read_text(encoding="utf-8")
    assert "<html" in html_text.lower()
    assert "</html>" in html_text.lower()
    assert flag_path.exists()


def test_conditional_retraining_logic(tmp_path: Path):
    """
    Requirement 13:
    Retraining only runs if drift_detected.flag exists.
    """
    flag_path = tmp_path / "drift_detected.flag"
    assert not flag_path.exists()

    # Step 1: Ensure retraining is aborted when flag does not exist
    assert is_retraining_triggered(flag_path) is False
    res_no_flag = execute_retraining(flag_path=flag_path, force=False)
    assert res_no_flag is None

    # Step 2: Create empty flag
    flag_path.touch()
    assert is_retraining_triggered(flag_path) is True
