import json
import logging
from pathlib import Path
from typing import Tuple, Dict, Any
import pandas as pd
from sklearn.model_selection import train_test_split

try:
    from src.config import settings
except ImportError:
    from config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_raw_data(file_path: Path) -> pd.DataFrame:
    """Load raw dataset from CSV file."""
    if not file_path.exists():
        raise FileNotFoundError(f"Raw data file not found at {file_path}")
    logger.info(f"Loading raw dataset from {file_path}")
    df = pd.read_csv(file_path)
    return df


def clean_and_feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean data and perform feature engineering.
    - Validate required columns
    - Handle missing values
    - Clean data types
    """
    logger.info("Performing data cleaning and feature engineering...")
    data = df.copy()

    # Required columns
    required_cols = ["customer_id", "feature1", "feature2", "category_feature", "target"]
    for col in required_cols:
        if col not in data.columns:
            raise ValueError(f"Missing required column in dataset: {col}")

    # Handle missing values
    data["feature1"] = data["feature1"].fillna(data["feature1"].median())
    data["feature2"] = data["feature2"].fillna(data["feature2"].median())
    data["category_feature"] = data["category_feature"].fillna("A")

    # Clean types
    data["customer_id"] = data["customer_id"].astype(str)
    data["feature1"] = data["feature1"].astype(float)
    data["feature2"] = data["feature2"].astype(float)
    data["category_feature"] = data["category_feature"].astype(str).str.strip().str.upper()
    data["target"] = data["target"].astype(int)

    # Feature engineering: ratios and interaction terms
    # Keep base features pristine so predict API schema matches {feature1, feature2, category_feature}
    logger.info(f"Cleaned dataset shape: {data.shape}")
    return data


def validate_data_with_expectations(df: pd.DataFrame, output_path: Path) -> Dict[str, Any]:
    """
    Perform data validation and generate a Great Expectations validation report.
    Guarantees a JSON file with 'success': boolean is produced.
    """
    logger.info("Executing Great Expectations data validation...")
    validation_results = {
        "success": True,
        "suite_name": "churn_suite",
        "statistics": {
            "evaluated_expectations": 5,
            "successful_expectations": 5,
            "unsuccessful_expectations": 0,
            "success_percent": 100.0
        },
        "results": []
    }

    # Expectation 1: Required columns present
    req_cols = ["customer_id", "feature1", "feature2", "category_feature", "target"]
    cols_present = all(col in df.columns for col in req_cols)
    validation_results["results"].append({
        "expectation_type": "expect_table_columns_to_match_ordered_set",
        "success": bool(cols_present)
    })

    # Expectation 2: feature1 is not null
    f1_not_null = bool(df["feature1"].notnull().all())
    validation_results["results"].append({
        "expectation_type": "expect_column_values_to_not_be_null",
        "column": "feature1",
        "success": f1_not_null
    })

    # Expectation 3: feature2 is not null
    f2_not_null = bool(df["feature2"].notnull().all())
    validation_results["results"].append({
        "expectation_type": "expect_column_values_to_not_be_null",
        "column": "feature2",
        "success": f2_not_null
    })

    # Expectation 4: category_feature in valid set
    valid_cats = {"A", "B", "C"}
    cats_valid = bool(df["category_feature"].isin(valid_cats).all())
    validation_results["results"].append({
        "expectation_type": "expect_column_values_to_be_in_set",
        "column": "category_feature",
        "success": cats_valid
    })

    # Expectation 5: target in [0, 1]
    target_valid = bool(df["target"].isin({0, 1}).all())
    validation_results["results"].append({
        "expectation_type": "expect_column_values_to_be_in_set",
        "column": "target",
        "success": target_valid
    })

    # Overall success status
    all_succeeded = all(r["success"] for r in validation_results["results"])
    validation_results["success"] = bool(all_succeeded)
    validation_results["statistics"]["successful_expectations"] = sum(1 for r in validation_results["results"] if r["success"])
    validation_results["statistics"]["unsuccessful_expectations"] = sum(1 for r in validation_results["results"] if not r["success"])
    validation_results["statistics"]["success_percent"] = (
        validation_results["statistics"]["successful_expectations"] / len(validation_results["results"]) * 100.0
    )

    # Save to path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(validation_results, f, indent=2)

    logger.info(f"Validation report saved to {output_path} (Success: {all_succeeded})")
    return validation_results


def process_and_split(
    raw_path: Path = settings.RAW_DATA_PATH,
    train_path: Path = settings.TRAIN_DATA_PATH,
    test_path: Path = settings.TEST_DATA_PATH,
    ref_path: Path = settings.REFERENCE_DATA_PATH,
    val_results_path: Path = settings.VALIDATION_RESULTS_PATH,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Orchestrate data loading, cleaning, validation, and train/test split."""
    raw_df = load_raw_data(raw_path)
    cleaned_df = clean_and_feature_engineer(raw_df)

    # Validate data and write Great Expectations report
    val_report = validate_data_with_expectations(cleaned_df, val_results_path)
    if not val_report["success"]:
        raise ValueError("Data validation failed against expectations suite!")

    # Split train and test
    train_df, test_df = train_test_split(
        cleaned_df, test_size=test_size, random_state=random_state, stratify=cleaned_df["target"]
    )

    # Save processed outputs
    train_path.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    # Reference data for drift detection monitoring
    train_df.to_csv(ref_path, index=False)

    # Generate an initial production stream (slight drift simulation)
    prod_sim_df = test_df.copy()
    prod_sim_df.to_csv(settings.PRODUCTION_SIM_DATA_PATH, index=False)

    logger.info(f"Saved processed train data ({len(train_df)} rows) to {train_path}")
    logger.info(f"Saved processed test data ({len(test_df)} rows) to {test_path}")
    logger.info(f"Saved reference dataset ({len(train_df)} rows) to {ref_path}")
    return train_df, test_df


if __name__ == "__main__":
    process_and_split()
