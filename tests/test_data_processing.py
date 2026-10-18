import json
from pathlib import Path
import pandas as pd
import pytest

from src.data_processing import (
    load_raw_data,
    clean_and_feature_engineer,
    validate_data_with_expectations,
    process_and_split
)


def test_load_raw_data(tmp_path: Path):
    csv_file = tmp_path / "test.csv"
    csv_file.write_text("customer_id,feature1,feature2,category_feature,target\nC1,1.0,2.0,A,0\n")
    df = load_raw_data(csv_file)
    assert len(df) == 1
    assert "customer_id" in df.columns

    with pytest.raises(FileNotFoundError):
        load_raw_data(tmp_path / "non_existent.csv")


def test_clean_and_feature_engineer():
    raw_df = pd.DataFrame({
        "customer_id": ["C1", "C2", "C3"],
        "feature1": [1.2, None, 3.4],
        "feature2": [None, 5.6, 7.8],
        "category_feature": ["A", " b ", None],
        "target": [0, 1, 0]
    })
    cleaned = clean_and_feature_engineer(raw_df)
    assert cleaned["feature1"].isnull().sum() == 0
    assert cleaned["feature2"].isnull().sum() == 0
    assert cleaned["category_feature"].tolist() == ["A", "B", "A"]
    assert cleaned["target"].dtype == int

    # Missing column raises ValueError
    bad_df = pd.DataFrame({"customer_id": ["C1"]})
    with pytest.raises(ValueError):
        clean_and_feature_engineer(bad_df)


def test_validate_data_with_expectations(tmp_path: Path):
    df = pd.DataFrame({
        "customer_id": ["C1", "C2"],
        "feature1": [1.0, 2.0],
        "feature2": [3.0, 4.0],
        "category_feature": ["A", "B"],
        "target": [0, 1]
    })
    out_json = tmp_path / "results.json"
    report = validate_data_with_expectations(df, out_json)

    assert out_json.exists()
    assert report["success"] is True
    assert isinstance(report["success"], bool)

    with open(out_json, "r") as f:
        data = json.load(f)
    assert "success" in data
    assert data["success"] is True


def test_process_and_split(tmp_path: Path):
    raw_csv = tmp_path / "raw.csv"
    rows = [
        {"customer_id": f"C{i}", "feature1": float(i), "feature2": float(i * 2), "category_feature": "A", "target": i % 2}
        for i in range(20)
    ]
    pd.DataFrame(rows).to_csv(raw_csv, index=False)

    train_csv = tmp_path / "train.csv"
    test_csv = tmp_path / "test.csv"
    ref_csv = tmp_path / "ref.csv"
    val_json = tmp_path / "val.json"

    train_df, test_df = process_and_split(
        raw_path=raw_csv,
        train_path=train_csv,
        test_path=test_csv,
        ref_path=ref_csv,
        val_results_path=val_json,
        test_size=0.25,
        random_state=42
    )
    assert len(train_df) == 15
    assert len(test_df) == 5
    assert train_csv.exists()
    assert test_csv.exists()
    assert val_json.exists()
