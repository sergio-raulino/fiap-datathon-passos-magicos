from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


TARGET_COLUMN = "risco_defasagem_futuro"
REGRESSION_TARGET_COLUMN = "defasagem_ano_seguinte"

FEATURE_COLUMNS = [
    "inde",
    "n_av",
    "iaa",
    "ieg",
    "ips",
    "ipp",
    "ida",
    "mat",
    "por",
    "ing",
    "ipv",
    "ian",
    "fase_ideal",
]

NUMERIC_FEATURES = [
    "inde",
    "n_av",
    "iaa",
    "ieg",
    "ips",
    "ipp",
    "ida",
    "mat",
    "por",
    "ing",
    "ipv",
    "ian",
]

CATEGORICAL_FEATURES = [
    "fase_ideal",
]


def _ensure_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    base = df.copy()

    for col in FEATURE_COLUMNS:
        if col not in base.columns:
            base[col] = np.nan

    return base[FEATURE_COLUMNS]


def _coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    base = df.copy()

    for col in NUMERIC_FEATURES:
        if col in base.columns:
            base[col] = pd.to_numeric(base[col], errors="coerce")

    return base


def _coerce_categorical_columns(df: pd.DataFrame) -> pd.DataFrame:
    base = df.copy()

    for col in CATEGORICAL_FEATURES:
        if col in base.columns:
            serie = base[col].astype("object")
            serie = serie.replace(
                {
                    "": np.nan,
                    "nan": np.nan,
                    "None": np.nan,
                    "<NA>": np.nan,
                }
            )
            serie = serie.where(pd.notna(serie), np.nan)
            base[col] = serie

    return base


def _replace_pd_na_with_np_nan(df: pd.DataFrame) -> pd.DataFrame:
    return df.where(pd.notna(df), np.nan)


def _build_inde_column(df: pd.DataFrame) -> pd.Series:
    inde = pd.Series(np.nan, index=df.index, dtype="float64")

    if "ano_pede" not in df.columns:
        return inde

    ano = pd.to_numeric(df["ano_pede"], errors="coerce")

    if "inde_2022" in df.columns:
        inde = inde.mask(ano == 2022, pd.to_numeric(df["inde_2022"], errors="coerce"))

    if "inde_2023" in df.columns:
        inde = inde.mask(ano == 2023, pd.to_numeric(df["inde_2023"], errors="coerce"))

    if "inde_2024" in df.columns:
        inde = inde.mask(ano == 2024, pd.to_numeric(df["inde_2024"], errors="coerce"))

    inde_fallback = (
        df.reindex(columns=["inde_2022", "inde_2023", "inde_2024"])
        .apply(pd.to_numeric, errors="coerce")
        .bfill(axis=1)
        .iloc[:, 0]
    )

    inde = inde.fillna(inde_fallback)

    return inde


def _build_future_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria alvos futuros por aluno (RA), usando o registro do ano seguinte.

    Nova definição de risco:
    - defasagem_ano_seguinte: valor da defasagem no próximo ano
    - risco_defasagem_futuro:
        1 => a defasagem piorou no ano seguinte
        0 => a defasagem permaneceu igual ou melhorou

    Regra:
    risco = 1 se defasagem_ano_seguinte > defasagem_atual
    """
    base = df.copy()

    if "ra" not in base.columns or "ano_pede" not in base.columns:
        base[REGRESSION_TARGET_COLUMN] = np.nan
        base[TARGET_COLUMN] = np.nan
        return base

    base["ano_pede"] = pd.to_numeric(base["ano_pede"], errors="coerce")
    base["defasagem"] = pd.to_numeric(base.get("defasagem"), errors="coerce")

    base = base.sort_values(["ra", "ano_pede"]).reset_index(drop=True)

    base["_proximo_ano"] = base.groupby("ra")["ano_pede"].shift(-1)
    base["_defasagem_proximo_ano"] = base.groupby("ra")["defasagem"].shift(-1)

    mask_next_year = base["_proximo_ano"] == (base["ano_pede"] + 1)

    base[REGRESSION_TARGET_COLUMN] = np.where(
        mask_next_year,
        base["_defasagem_proximo_ano"],
        np.nan,
    )

    base[TARGET_COLUMN] = np.where(
        pd.isna(base[REGRESSION_TARGET_COLUMN]) | pd.isna(base["defasagem"]),
        np.nan,
        np.where(
            (base[REGRESSION_TARGET_COLUMN] > 0) | (base["defasagem"] > 0),
            1,
            0,
        ),
    )

    base = base.drop(columns=["_proximo_ano", "_defasagem_proximo_ano"])

    return base


def create_model_base(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(
            columns=["ra"] + FEATURE_COLUMNS + [REGRESSION_TARGET_COLUMN, TARGET_COLUMN]
        )

    base = df.copy()

    base["inde"] = _build_inde_column(base)
    base = _build_future_targets(base)

    if "ra" not in base.columns:
        base["ra"] = pd.NA

    for col in FEATURE_COLUMNS:
        if col not in base.columns:
            base[col] = np.nan

    output_columns = ["ra"] + FEATURE_COLUMNS + [REGRESSION_TARGET_COLUMN, TARGET_COLUMN]
    model_df = base[output_columns].copy()

    model_df = _coerce_numeric_columns(model_df)

    if REGRESSION_TARGET_COLUMN in model_df.columns:
        model_df[REGRESSION_TARGET_COLUMN] = pd.to_numeric(
            model_df[REGRESSION_TARGET_COLUMN],
            errors="coerce",
        )

    if TARGET_COLUMN in model_df.columns:
        model_df[TARGET_COLUMN] = pd.to_numeric(
            model_df[TARGET_COLUMN],
            errors="coerce",
        )

    model_df = _coerce_categorical_columns(model_df)
    model_df = _replace_pd_na_with_np_nan(model_df)

    if TARGET_COLUMN in model_df.columns:
        model_df[TARGET_COLUMN] = model_df[TARGET_COLUMN].astype("Int64")

    return model_df


def prepare_training_features(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
) -> tuple[pd.DataFrame, pd.Series]:
    if target_column not in df.columns:
        raise ValueError(
            f"A coluna alvo '{target_column}' não foi encontrada na base de treino."
        )

    X = _ensure_feature_columns(df)
    X = _coerce_numeric_columns(X)
    X = _coerce_categorical_columns(X)
    X = _replace_pd_na_with_np_nan(X)

    y = pd.to_numeric(df[target_column], errors="coerce")
    valid_mask = y.notna()

    X = X.loc[valid_mask].reset_index(drop=True)
    y = y.loc[valid_mask].astype(int).reset_index(drop=True)

    return X, y


def prepare_regression_training_features(
    df: pd.DataFrame,
    target_column: str = REGRESSION_TARGET_COLUMN,
) -> tuple[pd.DataFrame, pd.Series]:
    if target_column not in df.columns:
        raise ValueError(
            f"A coluna alvo '{target_column}' não foi encontrada na base de treino."
        )

    X = _ensure_feature_columns(df)
    X = _coerce_numeric_columns(X)
    X = _coerce_categorical_columns(X)
    X = _replace_pd_na_with_np_nan(X)

    y = pd.to_numeric(df[target_column], errors="coerce")
    valid_mask = y.notna()

    X = X.loc[valid_mask].reset_index(drop=True)
    y = y.loc[valid_mask].reset_index(drop=True)

    return X, y


def prepare_inference_features(
    df: pd.DataFrame,
    artifacts: dict[str, Any] | None = None,
) -> pd.DataFrame:
    base = _ensure_feature_columns(df)
    base = _coerce_numeric_columns(base)
    base = _coerce_categorical_columns(base)
    base = _replace_pd_na_with_np_nan(base)

    if artifacts and isinstance(artifacts, dict):
        expected_cols = artifacts.get("feature_columns")
        if expected_cols:
            for col in expected_cols:
                if col not in base.columns:
                    base[col] = np.nan
            base = base[expected_cols]

    return base


def build_inference_features(
    df: pd.DataFrame,
    artifacts: dict[str, Any] | None = None,
) -> pd.DataFrame:
    return prepare_inference_features(df, artifacts=artifacts)


def transform_features_for_inference(
    df: pd.DataFrame,
    artifacts: dict[str, Any] | None = None,
) -> pd.DataFrame:
    return prepare_inference_features(df, artifacts=artifacts)