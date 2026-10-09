from pathlib import Path

from ucimlrepo import fetch_ucirepo


DATASET_ID = 296


def main() -> None:
    project_root = Path(__file__).resolve().parents[3]

    raw_data_path = (
        project_root / "data" / "raw" / "diabetic_data.csv"
    )
    dictionary_path = (
        project_root / "docs" / "uci_variables.csv"
    )

    raw_data_path.parent.mkdir(parents=True, exist_ok=True)
    dictionary_path.parent.mkdir(parents=True, exist_ok=True)

    print("Загрузка датасета UCI...")

    dataset = fetch_ucirepo(id=DATASET_ID)
    data = dataset.data.original.copy()

    data.to_csv(raw_data_path, index=False)
    dataset.variables.to_csv(dictionary_path, index=False)

    print(f"Данные сохранены: {raw_data_path}")
    print(f"Словарь признаков сохранён: {dictionary_path}")
    print(f"Размер датасета: {data.shape[0]} строк, {data.shape[1]} столбцов")

    if "readmitted" in data.columns:
        print("\nРаспределение целевой переменной:")
        print(data["readmitted"].value_counts(dropna=False))


if __name__ == "__main__":
    main()