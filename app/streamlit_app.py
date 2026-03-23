from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from src import clean as clean_mod
except Exception:
    clean_mod = None

try:
    from src import features as features_mod
except Exception:
    features_mod = None

try:
    from src import load as load_mod
except Exception:
    load_mod = None

try:
    from src import model as model_mod
except Exception:
    model_mod = None

try:
    from src import unify as unify_mod
except Exception:
    unify_mod = None


st.set_page_config(
    page_title="Passos Mágicos • Solução Preditiva",
    page_icon="✨",
    layout="wide",
)


DEFAULT_ANALYTIC_PARQUET = ROOT_DIR / "data" / "processed" / "base_PEDE_consolidada_analitica.parquet"


def resolve_callable(module: Any, candidates: list[str]) -> Callable | None:
    if module is None:
        return None
    for name in candidates:
        fn = getattr(module, name, None)
        if callable(fn):
            return fn
    return None


def normalize_scalar(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


def df_to_download_bytes(df: pd.DataFrame, filetype: str = "csv") -> bytes:
    if filetype == "xlsx":
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="predicoes")
        output.seek(0)
        return output.read()
    return df.to_csv(index=False).encode("utf-8-sig")


clean_dataframe_fn = resolve_callable(
    clean_mod,
    ["clean_dataframe", "clean_df", "apply_cleaning", "clean_dataset"],
)

prepare_inference_features_fn = resolve_callable(
    features_mod,
    [
        "prepare_inference_features",
        "build_inference_features",
        "transform_features_for_inference",
        "make_inference_features",
        "prepare_features",
    ],
)

load_model_fn = resolve_callable(
    model_mod,
    ["load_model", "load_artifacts", "get_model"],
)

predict_fn = resolve_callable(
    model_mod,
    ["predict_dataframe", "predict", "run_inference", "infer"],
)

predict_proba_fn = resolve_callable(
    model_mod,
    ["predict_proba_dataframe", "predict_proba", "run_predict_proba"],
)


@st.cache_resource(show_spinner=False)
def load_model_resource():
    if load_model_fn is None:
        return None
    return load_model_fn()


@st.cache_data(show_spinner=False)
def load_analytic_base(parquet_path: str) -> pd.DataFrame:
    path = Path(parquet_path)

    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")

    suffix = path.suffix.lower()
    if suffix == ".parquet":
        df = pd.read_parquet(path)
    elif suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        raise ValueError(f"Formato não suportado: {suffix}")

    if clean_dataframe_fn is not None:
        try:
            df = clean_dataframe_fn(df)
        except Exception:
            pass

    return df


def model_bundle_to_model(model_bundle: Any) -> Any:
    if model_bundle is None:
        return None
    if isinstance(model_bundle, dict):
        if "model" in model_bundle:
            return model_bundle["model"]
        if "estimator" in model_bundle:
            return model_bundle["estimator"]
    return model_bundle


def apply_feature_pipeline(input_df: pd.DataFrame, model_bundle: Any) -> pd.DataFrame:
    if prepare_inference_features_fn is None:
        raise RuntimeError(
            "Não encontrei função de inferência em src/features.py. "
            "Crie prepare_inference_features(df, artifacts=None)."
        )

    try:
        return prepare_inference_features_fn(input_df, artifacts=model_bundle)
    except TypeError:
        try:
            return prepare_inference_features_fn(input_df, model_bundle)
        except TypeError:
            return prepare_inference_features_fn(input_df)


def run_prediction_pipeline(input_df: pd.DataFrame) -> pd.DataFrame:
    model_bundle = load_model_resource()
    if model_bundle is None:
        raise RuntimeError(
            "Não encontrei load_model() em src/model.py."
        )

    if clean_dataframe_fn is not None:
        try:
            input_df = clean_dataframe_fn(input_df)
        except Exception:
            pass

    X = apply_feature_pipeline(input_df.copy(), model_bundle)
    model_obj = model_bundle_to_model(model_bundle)

    if predict_fn is not None:
        try:
            preds = predict_fn(X, model=model_obj, artifacts=model_bundle)
        except TypeError:
            try:
                preds = predict_fn(X, model_obj)
            except TypeError:
                preds = predict_fn(X)
    else:
        preds = model_obj.predict(X)

    result = input_df.copy()
    result["predicao"] = preds

    if predict_proba_fn is not None:
        try:
            proba = predict_proba_fn(X, model=model_obj, artifacts=model_bundle)
        except TypeError:
            try:
                proba = predict_proba_fn(X, model_obj)
            except TypeError:
                proba = predict_proba_fn(X)
    elif hasattr(model_obj, "predict_proba"):
        proba = model_obj.predict_proba(X)
    else:
        proba = None

    if proba is not None:
        proba_df = pd.DataFrame(proba)
        if proba_df.shape[1] == 2:
            result["probabilidade_classe_0"] = proba_df.iloc[:, 0]
            result["probabilidade_classe_1"] = proba_df.iloc[:, 1]
        else:
            for idx in range(proba_df.shape[1]):
                result[f"probabilidade_classe_{idx}"] = proba_df.iloc[:, idx]

    return result


def render_sidebar() -> dict[str, Any]:
    st.sidebar.title("Configurações")

    analytic_file = st.sidebar.text_input(
        "Base analítica consolidada",
        value=str(DEFAULT_ANALYTIC_PARQUET),
    )

    st.sidebar.markdown("---")
    st.sidebar.caption("A aplicação usa o modelo treinado em `data/artifacts/`.")

    return {
        "analytic_file": analytic_file,
    }


def render_project_status():
    st.subheader("Status técnico")

    rows = [
        {
            "módulo": "src/features.py",
            "função": prepare_inference_features_fn.__name__ if prepare_inference_features_fn else "não encontrada",
            "status": "OK" if prepare_inference_features_fn else "ajustar",
        },
        {
            "módulo": "src/model.py (load)",
            "função": load_model_fn.__name__ if load_model_fn else "não encontrada",
            "status": "OK" if load_model_fn else "ajustar",
        },
        {
            "módulo": "src/model.py (predict)",
            "função": predict_fn.__name__ if predict_fn else "fallback .predict()",
            "status": "OK",
        },
        {
            "módulo": "src/clean.py",
            "função": clean_dataframe_fn.__name__ if clean_dataframe_fn else "opcional",
            "status": "OK" if clean_dataframe_fn else "opcional",
        },
    ]

    st.dataframe(pd.DataFrame(rows), use_container_width=True)


def render_single_prediction():
    st.subheader("Predição individual")

    with st.form("predicao_individual"):
        col1, col2, col3 = st.columns(3)

        with col1:
            ano_pede = st.number_input("Ano PEDE", min_value=2022, max_value=2035, value=2024)
            inde = st.number_input("INDE", min_value=0.0, max_value=10.0, value=7.0, step=0.1)
            n_av = st.number_input("Nº de avaliações", min_value=0.0, max_value=20.0, value=6.0, step=1.0)
            iaa = st.number_input("IAA", min_value=0.0, max_value=10.0, value=7.0, step=0.1)
            ieg = st.number_input("IEG", min_value=0.0, max_value=10.0, value=8.0, step=0.1)

        with col2:
            ips = st.number_input("IPS", min_value=0.0, max_value=10.0, value=7.0, step=0.1)
            ipp = st.number_input("IPP", min_value=0.0, max_value=10.0, value=7.0, step=0.1)
            ida = st.number_input("IDA", min_value=0.0, max_value=10.0, value=6.0, step=0.1)
            mat = st.number_input("Matemática", min_value=0.0, max_value=10.0, value=6.0, step=0.1)
            por = st.number_input("Português", min_value=0.0, max_value=10.0, value=6.0, step=0.1)

        with col3:
            ing = st.number_input("Inglês", min_value=0.0, max_value=10.0, value=6.0, step=0.1)
            ipv = st.number_input("IPV", min_value=0.0, max_value=10.0, value=7.0, step=0.1)
            ian = st.number_input("IAN", min_value=0.0, max_value=10.0, value=7.0, step=0.1)
            fase_ideal = st.selectbox("Fase ideal", options=[str(i) for i in range(10)], index=1)
            defasagem = st.number_input("Defasagem", min_value=-5.0, max_value=10.0, value=0.0, step=1.0)

        submitted = st.form_submit_button("Gerar predição")

    if not submitted:
        return

    input_df = pd.DataFrame(
        [
            {
                "ano_pede": ano_pede,
                "inde": inde,
                "n_av": n_av,
                "iaa": iaa,
                "ieg": ieg,
                "ips": ips,
                "ipp": ipp,
                "ida": ida,
                "mat": mat,
                "por": por,
                "ing": ing,
                "ipv": ipv,
                "ian": ian,
                "fase_ideal": fase_ideal,
                "defasagem": defasagem,
            }
        ]
    )

    try:
        result_df = run_prediction_pipeline(input_df)
    except Exception as exc:
        st.error(f"Erro ao gerar predição: {exc}")
        st.stop()

    row = result_df.iloc[0].to_dict()

    st.success("Predição realizada com sucesso.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Predição", normalize_scalar(row.get("predicao")))

    with col2:
        if "probabilidade_classe_1" in row:
            st.metric("Prob. classe 1", f"{float(row['probabilidade_classe_1']):.2%}")

    with col3:
        if "probabilidade_classe_0" in row:
            st.metric("Prob. classe 0", f"{float(row['probabilidade_classe_0']):.2%}")

    with st.expander("Ver dados usados"):
        st.dataframe(result_df, use_container_width=True)


def render_batch_prediction():
    st.subheader("Predição em lote")

    st.caption(
        "Envie um CSV, XLSX ou Parquet contendo as colunas do modelo: "
        "ano_pede, inde, n_av, iaa, ieg, ips, ipp, ida, mat, por, ing, ipv, ian, fase_ideal, defasagem."
    )

    uploaded_file = st.file_uploader(
        "Arquivo para predição em lote",
        type=["csv", "xlsx", "parquet"],
        key="batch_upload",
    )

    if uploaded_file is None:
        st.info("Envie um arquivo para processar.")
        return

    try:
        name = uploaded_file.name.lower()
        if name.endswith(".csv"):
            batch_df = pd.read_csv(uploaded_file)
        elif name.endswith(".xlsx"):
            batch_df = pd.read_excel(uploaded_file)
        elif name.endswith(".parquet"):
            batch_df = pd.read_parquet(uploaded_file)
        else:
            raise ValueError("Formato não suportado.")
    except Exception as exc:
        st.error(f"Não foi possível ler o arquivo: {exc}")
        return

    st.write("Prévia do arquivo")
    st.dataframe(batch_df.head(20), use_container_width=True)

    if st.button("Executar predição em lote"):
        try:
            result_df = run_prediction_pipeline(batch_df)
        except Exception as exc:
            st.error(f"Erro ao executar inferência: {exc}")
            return

        st.success(f"Predição concluída para {len(result_df)} registro(s).")
        st.dataframe(result_df.head(50), use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "Baixar CSV",
                data=df_to_download_bytes(result_df, "csv"),
                file_name="predicoes_passos_magicos.csv",
                mime="text/csv",
            )
        with col2:
            st.download_button(
                "Baixar XLSX",
                data=df_to_download_bytes(result_df, "xlsx"),
                file_name="predicoes_passos_magicos.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )


def render_data_explorer(analytic_file: str):
    st.subheader("Base consolidada analítica")

    analytic_path = Path(analytic_file)

    if not analytic_path.exists():
        st.warning("O arquivo informado não existe.")
        return

    try:
        df = load_analytic_base(str(analytic_path))
    except Exception as exc:
        st.error(f"Erro ao carregar a base: {exc}")
        return

    if df.empty:
        st.info("A base está vazia.")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Linhas", f"{len(df):,}".replace(",", "."))
    c2.metric("Colunas", df.shape[1])
    c3.metric("Anos", ", ".join(map(str, sorted(df['ano_pede'].dropna().unique()))) if "ano_pede" in df.columns else "-")

    st.dataframe(df.head(50), use_container_width=True)

    with st.expander("Distribuições"):
        if "fase" in df.columns:
            fase_dist = (
                df["fase"]
                .fillna("NA")
                .astype(str)
                .value_counts()
                .rename_axis("fase")
                .reset_index(name="quantidade")
            )
            st.write("Distribuição de fase")
            st.dataframe(fase_dist, use_container_width=True)

        if "turma" in df.columns:
            turma_dist = (
                df["turma"]
                .fillna("NA")
                .astype(str)
                .value_counts()
                .rename_axis("turma")
                .reset_index(name="quantidade")
            )
            st.write("Distribuição de turma")
            st.dataframe(turma_dist, use_container_width=True)


def main():
    cfg = render_sidebar()

    st.title("✨ Passos Mágicos • Solução Preditiva")
    st.caption(
        "Aplicação em Streamlit para disponibilização do modelo preditivo "
        "treinado a partir da base processada reduzida de ML."
    )

    tabs = st.tabs(
        [
            "Predição individual",
            "Predição em lote",
            "Base consolidada",
            "Status técnico",
        ]
    )

    with tabs[0]:
        render_single_prediction()

    with tabs[1]:
        render_batch_prediction()

    with tabs[2]:
        render_data_explorer(cfg["analytic_file"])

    with tabs[3]:
        render_project_status()


if __name__ == "__main__":
    main()