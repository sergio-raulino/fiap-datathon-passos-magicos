from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    prepare_inference_features,
    prepare_training_features,
)

ROOT_DIR = Path(__file__).resolve().parents[1]

DEFAULT_TRAINING_FILE = ROOT_DIR / "data" / "processed" / "base_processada_reduzida_ML.parquet"
ARTIFACTS_DIR = ROOT_DIR / "data" / "artifacts"

MODEL_FILE = ARTIFACTS_DIR / "model_pipeline.joblib"
METADATA_FILE = ARTIFACTS_DIR / "model_metadata.joblib"


def load_training_data(processed_path: str | Path = DEFAULT_TRAINING_FILE) -> pd.DataFrame:
    processed_path = Path(processed_path)

    if not processed_path.exists():
        raise FileNotFoundError(f"Arquivo de treino não encontrado: {processed_path}")

    suffix = processed_path.suffix.lower()

    if suffix == ".parquet":
        return pd.read_parquet(processed_path)
    if suffix == ".csv":
        return pd.read_csv(processed_path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(processed_path)

    raise ValueError(f"Formato de arquivo não suportado para treino: {processed_path.suffix}")


def build_pipeline() -> Pipeline:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )

    estimator = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", estimator),
        ]
    )


def train_model(
    processed_path: str | Path = DEFAULT_TRAINING_FILE,
    target_column: str = TARGET_COLUMN,
    test_size: float = 0.2,
    random_state: int = 42,
    save_artifacts: bool = True,
) -> dict[str, Any]:
    print("Arquivo de treino:", processed_path)

    df = load_training_data(processed_path)
    X, y = prepare_training_features(df, target_column=target_column)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if y.nunique() > 1 else None,
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    try:
        print("Classes do modelo:", pipeline.named_steps["model"].classes_)
    except Exception:
        print("Não foi possível ler classes_")

    y_pred = pipeline.predict(X_test)

    metrics: dict[str, Any] = {
        "classification_report": classification_report(
            y_test,
            y_pred,
            output_dict=True,
            zero_division=0,
        ),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "target_column": target_column,
        "feature_columns": FEATURE_COLUMNS,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
    }

    if y.nunique() == 2 and hasattr(pipeline, "predict_proba"):
        y_score = pipeline.predict_proba(X_test)[:, 1]
        metrics["roc_auc"] = float(roc_auc_score(y_test, y_score))

    artifacts = {
        "model": pipeline,
        "pipeline": pipeline,
        "feature_columns": FEATURE_COLUMNS,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "target_column": target_column,
        "metrics": metrics,
        "artifacts_dir": str(ARTIFACTS_DIR),
        "model_path": str(MODEL_FILE),
        "metadata_path": str(METADATA_FILE),
    }

    if save_artifacts:
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

        joblib.dump(pipeline, MODEL_FILE)
        joblib.dump(
            {
                "feature_columns": FEATURE_COLUMNS,
                "numeric_features": NUMERIC_FEATURES,
                "categorical_features": CATEGORICAL_FEATURES,
                "target_column": target_column,
                "metrics": metrics,
            },
            METADATA_FILE,
        )

        print(f"Pipeline salvo em: {MODEL_FILE}")
        print(f"Metadados salvos em: {METADATA_FILE}")

    return artifacts


def _resolve_model_file() -> Path:
    candidates = [
        MODEL_FILE,
        ARTIFACTS_DIR / "model.joblib",
        ARTIFACTS_DIR / "pipeline.joblib",
        ARTIFACTS_DIR / "model.pkl",
        ARTIFACTS_DIR / "pipeline.pkl",
        ARTIFACTS_DIR / "artifacts.pkl",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    available_files = []
    if ARTIFACTS_DIR.exists():
        available_files = sorted([p.name for p in ARTIFACTS_DIR.iterdir() if p.is_file()])

    raise FileNotFoundError(
        f"Não encontrei artefatos do modelo em {ARTIFACTS_DIR}. "
        "Esperado algo como model_pipeline.joblib, model.joblib, pipeline.joblib, "
        "model.pkl, pipeline.pkl ou artifacts.pkl. "
        f"Arquivos encontrados: {available_files}"
    )


def load_model() -> dict[str, Any]:
    if not ARTIFACTS_DIR.exists():
        raise FileNotFoundError(
            f"Diretório de artefatos não encontrado: {ARTIFACTS_DIR}. "
            "Verifique se a pasta data/artifacts foi enviada ao repositório."
        )

    resolved_model_file = _resolve_model_file()
    pipeline = joblib.load(resolved_model_file)

    metadata: dict[str, Any] = {
        "feature_columns": FEATURE_COLUMNS,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "target_column": TARGET_COLUMN,
        "metrics": {},
    }

    if METADATA_FILE.exists():
        loaded_metadata = joblib.load(METADATA_FILE)
        if isinstance(loaded_metadata, dict):
            metadata.update(loaded_metadata)

    return {
        "model": pipeline,
        "pipeline": pipeline,
        "feature_columns": metadata["feature_columns"],
        "numeric_features": metadata["numeric_features"],
        "categorical_features": metadata["categorical_features"],
        "target_column": metadata["target_column"],
        "metrics": metadata.get("metrics", {}),
        "artifacts_dir": str(ARTIFACTS_DIR),
        "model_path": str(resolved_model_file),
        "metadata_path": str(METADATA_FILE),
    }


def predict_dataframe(
    X: pd.DataFrame,
    model: Any = None,
    artifacts: dict[str, Any] | None = None,
):
    if artifacts is None:
        artifacts = load_model()

    if model is None:
        model = artifacts.get("model")

    if model is None:
        raise ValueError("Nenhum modelo foi carregado para predição.")

    X_prepared = prepare_inference_features(X, artifacts=artifacts)
    return model.predict(X_prepared)


def predict_proba_dataframe(
    X: pd.DataFrame,
    model: Any = None,
    artifacts: dict[str, Any] | None = None,
):
    if artifacts is None:
        artifacts = load_model()

    if model is None:
        model = artifacts.get("model")

    if model is None:
        raise ValueError("Nenhum modelo foi carregado para predição de probabilidades.")

    if not hasattr(model, "predict_proba"):
        raise AttributeError("O modelo carregado não suporta predict_proba().")

    X_prepared = prepare_inference_features(X, artifacts=artifacts)
    return model.predict_proba(X_prepared)


def run_inference(df: pd.DataFrame) -> pd.DataFrame:
    artifacts = load_model()
    model = artifacts["model"]

    X_prepared = prepare_inference_features(df, artifacts=artifacts)

    result = df.copy()
    result["predicao"] = model.predict(X_prepared)

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_prepared)
        proba_df = pd.DataFrame(proba)

        if proba_df.shape[1] == 2:
            result["probabilidade_classe_0"] = proba_df.iloc[:, 0]
            result["probabilidade_classe_1"] = proba_df.iloc[:, 1]
        else:
            for idx in range(proba_df.shape[1]):
                result[f"probabilidade_classe_{idx}"] = proba_df.iloc[:, idx]

    return result