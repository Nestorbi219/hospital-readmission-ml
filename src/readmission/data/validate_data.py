from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "diabetic_data.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"

REQUIRED_COLUMNS = {
    "encounter_id",
    "patient_nbr",
    "readmitted",
}


def main() -> None:
    data = pd.read_csv(DATA_PATH, low_memory=False)

    missing_required = REQUIRED_COLUMNS - set(data.columns)
    if missing_required:
        raise ValueError(
            f"Отсутствуют обязательные столбцы: {sorted(missing_required)}"
        )

    # В исходном датасете символ ? обозначает пропуск.
    normalized_data = data.replace("?", pd.NA)

    patient_counts = data["patient_nbr"].value_counts()
    target_counts = data["readmitted"].value_counts(dropna=False)

    constant_columns = [
        column
        for column in data.columns
        if data[column].nunique(dropna=False) <= 1
    ]

    summary = {
        "rows": len(data),
        "columns": data.shape[1],
        "exact_duplicate_rows": int(data.duplicated().sum()),
        "duplicate_encounter_ids": int(
            data["encounter_id"].duplicated().sum()
        ),
        "unique_patients": int(data["patient_nbr"].nunique()),
        "patients_with_multiple_encounters": int(
            (patient_counts > 1).sum()
        ),
        "positive_target_count": int(
            (data["readmitted"] == "<30").sum()
        ),
        "positive_target_rate_pct": round(
            (data["readmitted"] == "<30").mean() * 100,
            2,
        ),
        "constant_columns_count": len(constant_columns),
        "constant_columns": ", ".join(constant_columns),
    }

    summary_table = pd.DataFrame(
        summary.items(),
        columns=["metric", "value"],
    )

    missingness = pd.DataFrame(
        {
            "column": normalized_data.columns,
            "missing_count": normalized_data.isna().sum().values,
            "missing_pct": (
                normalized_data.isna().mean().values * 100
            ).round(2),
        }
    )
    missingness = missingness[
        missingness["missing_count"] > 0
    ].sort_values("missing_pct", ascending=False)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_table.to_csv(
        REPORTS_DIR / "data_quality_summary.csv",
        index=False,
    )
    missingness.to_csv(
        REPORTS_DIR / "missingness.csv",
        index=False,
    )

    print("\nОсновные показатели:")
    print(summary_table.to_string(index=False))

    print("\nРаспределение readmitted:")
    print(target_counts.to_string())

    print("\nСтолбцы с наибольшей долей пропусков:")
    print(missingness.head(10).to_string(index=False))

    print("\nОтчёты сохранены в папку reports/")


if __name__ == "__main__":
    main()