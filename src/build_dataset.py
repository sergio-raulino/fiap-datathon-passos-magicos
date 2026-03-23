from pathlib import Path

from src.unify import build_analytic_base
from src.features import create_model_base


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent

    raw_file = project_root / "data" / "raw" / "BASE DE DADOS PEDE 2024 - DATATHON.xlsx"
    processed_dir = project_root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    df_analytic = build_analytic_base(raw_file)
    df_model = create_model_base(df_analytic)

    df_analytic.to_excel(
        processed_dir / "Base de Dados PEDE - Consolidada Analitica.xlsx",
        index=False,
    )
    df_analytic.to_parquet(
        processed_dir / "base_PEDE_consolidada_analitica.parquet",
        index=False,
    )

    df_model.to_excel(
        processed_dir / "base_processada_reduzida_ML.xlsx",
        index=False,
    )
    df_model.to_parquet(
        processed_dir / "base_processada_reduzida_ML.parquet",
        index=False,
    )

    print("Bases geradas com sucesso em data/processed/")


if __name__ == "__main__":
    main()