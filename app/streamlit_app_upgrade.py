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
    from src import model as model_mod
except Exception:
    model_mod = None


st.set_page_config(
    page_title="Passos Mágicos • Solução Preditiva",
    page_icon="✨",
    layout="wide",
)

DEFAULT_ANALYTIC_PARQUET = ROOT_DIR / "data" / "processed" / "base_PEDE_consolidada_analitica.parquet"


# =========================================================
# Helpers
# =========================================================
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


def classify_risk(prob_classe_1: float | None) -> str:
    if prob_classe_1 is None:
        return "Indisponível"
    if prob_classe_1 < 0.30:
        return "Baixo"
    if prob_classe_1 < 0.70:
        return "Moderado"
    return "Alto"


def prediction_label(pred: Any) -> str:
    if pred in [1, "1", True]:
        return "Com risco de defasagem futura"
    return "Sem risco de defasagem futura"


def build_prediction_explanation(input_row: pd.Series, prob_classe_1: float | None) -> list[str]:
    reasons: list[str] = []

    def val(col: str, default: float = 0.0) -> float:
        x = input_row.get(col, default)
        try:
            return float(x)
        except Exception:
            return default

    if val("defasagem") > 0:
        reasons.append("O aluno já apresenta defasagem no ano atual, o que aumenta a chance de continuidade no ano seguinte.")

    if val("inde") < 6:
        reasons.append("O INDE informado está abaixo de 6, sugerindo desempenho global mais vulnerável.")

    low_subjects = []
    if val("mat") < 6:
        low_subjects.append("Matemática")
    if val("por") < 6:
        low_subjects.append("Português")
    if val("ing") < 6:
        low_subjects.append("Inglês")
    if low_subjects:
        reasons.append(f"Há desempenho mais baixo em disciplinas-chave: {', '.join(low_subjects)}.")

    low_indexes = []
    for col, label in [("iaa", "IAA"), ("ieg", "IEG"), ("ips", "IPS"), ("ipp", "IPP"), ("ida", "IDA"), ("ipv", "IPV"), ("ian", "IAN")]:
        if val(col) < 6:
            low_indexes.append(label)
    if low_indexes:
        reasons.append(f"Alguns indicadores complementares estão abaixo de 6: {', '.join(low_indexes)}.")

    if val("n_av") < 5:
        reasons.append("O número de avaliações é relativamente baixo, o que pode refletir menor robustez de acompanhamento.")

    if prob_classe_1 is not None:
        level = classify_risk(prob_classe_1).lower()
        reasons.append(f"A probabilidade estimada pelo modelo foi classificada como risco {level}.")

    if not reasons:
        reasons.append("Os indicadores informados estão em faixa favorável, o que contribui para uma previsão de menor risco.")

    return reasons


def format_prediction_output(result_df: pd.DataFrame) -> pd.DataFrame:
    output = result_df.copy()

    if "predicao" in output.columns:
        output["resultado_predicao"] = output["predicao"].apply(prediction_label)

    if "probabilidade_classe_1" in output.columns:
        output["risco_estimado"] = output["probabilidade_classe_1"].apply(classify_risk)
        output["probabilidade_classe_1_pct"] = output["probabilidade_classe_1"].apply(lambda x: f"{float(x):.2%}")

    if "probabilidade_classe_0" in output.columns:
        output["probabilidade_classe_0_pct"] = output["probabilidade_classe_0"].apply(lambda x: f"{float(x):.2%}")

    return output


# =========================================================
# Dynamic resolution
# =========================================================
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


# =========================================================
# Cached loading
# =========================================================
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


# =========================================================
# Model execution
# =========================================================
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
        raise RuntimeError("Não encontrei load_model() em src/model.py.")

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


# =========================================================
# UI
# =========================================================
def render_sidebar() -> dict[str, Any]:
    st.sidebar.title("Configurações")

    analytic_file = st.sidebar.text_input(
        "Base analítica consolidada",
        value=str(DEFAULT_ANALYTIC_PARQUET),
    )

    st.sidebar.markdown("---")
    st.sidebar.caption("A aplicação utiliza o modelo treinado salvo em `data/artifacts/`.")

    st.sidebar.markdown("### Cenários rápidos")
    scenario = st.sidebar.selectbox(
        "Selecione um exemplo para preencher na predição individual",
        options=[
            "Nenhum",
            "Aluno com baixo risco",
            "Aluno com risco moderado",
            "Aluno com alto risco",
        ],
        index=0,
    )

    return {
        "analytic_file": analytic_file,
        "scenario": scenario,
    }


def get_default_values_from_scenario(scenario: str) -> dict[str, Any]:
    base = {
        "ano_pede": 2023,
        "inde": 7.0,
        "n_av": 6.0,
        "iaa": 7.0,
        "ieg": 8.0,
        "ips": 7.0,
        "ipp": 7.0,
        "ida": 6.0,
        "mat": 6.0,
        "por": 6.0,
        "ing": 6.0,
        "ipv": 7.0,
        "ian": 7.0,
        "fase_ideal": "1",
        "defasagem": 0.0,
    }

    if scenario == "Aluno com baixo risco":
        return {
            **base,
            "inde": 8.5,
            "n_av": 7.0,
            "iaa": 8.0,
            "ieg": 8.0,
            "ips": 8.0,
            "ipp": 8.0,
            "ida": 8.0,
            "mat": 8.0,
            "por": 8.0,
            "ing": 8.0,
            "ipv": 8.0,
            "ian": 8.0,
            "fase_ideal": "7",
            "defasagem": 0.0,
        }

    if scenario == "Aluno com risco moderado":
        return {
            **base,
            "inde": 6.0,
            "n_av": 5.0,
            "iaa": 6.0,
            "ieg": 6.0,
            "ips": 6.0,
            "ipp": 6.0,
            "ida": 6.0,
            "mat": 5.0,
            "por": 6.0,
            "ing": 5.0,
            "ipv": 6.0,
            "ian": 6.0,
            "fase_ideal": "5",
            "defasagem": 1.0,
        }

    if scenario == "Aluno com alto risco":
        return {
            **base,
            "inde": 4.0,
            "n_av": 3.0,
            "iaa": 4.0,
            "ieg": 4.0,
            "ips": 4.0,
            "ipp": 4.0,
            "ida": 4.0,
            "mat": 3.0,
            "por": 4.0,
            "ing": 3.0,
            "ipv": 4.0,
            "ian": 4.0,
            "fase_ideal": "4",
            "defasagem": 2.0,
        }

    return base


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


def render_single_prediction(defaults: dict[str, Any]):
    st.subheader("Predição individual")
    st.caption(
        "Informe os indicadores atuais do aluno para estimar a probabilidade de "
        "defasagem no ano seguinte."
    )

    with st.form("predicao_individual"):
        col1, col2, col3 = st.columns(3)

        with col1:
            ano_pede = st.number_input("Ano PEDE", min_value=2022, max_value=2035, value=int(defaults["ano_pede"]))
            inde = st.number_input("INDE", min_value=0.0, max_value=10.0, value=float(defaults["inde"]), step=0.1)
            n_av = st.number_input("Nº de avaliações", min_value=0.0, max_value=20.0, value=float(defaults["n_av"]), step=1.0)
            iaa = st.number_input("IAA", min_value=0.0, max_value=10.0, value=float(defaults["iaa"]), step=0.1)
            ieg = st.number_input("IEG", min_value=0.0, max_value=10.0, value=float(defaults["ieg"]), step=0.1)

        with col2:
            ips = st.number_input("IPS", min_value=0.0, max_value=10.0, value=float(defaults["ips"]), step=0.1)
            ipp = st.number_input("IPP", min_value=0.0, max_value=10.0, value=float(defaults["ipp"]), step=0.1)
            ida = st.number_input("IDA", min_value=0.0, max_value=10.0, value=float(defaults["ida"]), step=0.1)
            mat = st.number_input("Matemática", min_value=0.0, max_value=10.0, value=float(defaults["mat"]), step=0.1)
            por = st.number_input("Português", min_value=0.0, max_value=10.0, value=float(defaults["por"]), step=0.1)

        with col3:
            ing = st.number_input("Inglês", min_value=0.0, max_value=10.0, value=float(defaults["ing"]), step=0.1)
            ipv = st.number_input("IPV", min_value=0.0, max_value=10.0, value=float(defaults["ipv"]), step=0.1)
            ian = st.number_input("IAN", min_value=0.0, max_value=10.0, value=float(defaults["ian"]), step=0.1)
            fase_options = [str(i) for i in range(10)]
            fase_index = fase_options.index(str(defaults["fase_ideal"])) if str(defaults["fase_ideal"]) in fase_options else 1
            fase_ideal = st.selectbox("Fase ideal", options=fase_options, index=fase_index)
            defasagem = st.number_input("Defasagem atual", min_value=-5.0, max_value=10.0, value=float(defaults["defasagem"]), step=1.0)

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
    pred = normalize_scalar(row.get("predicao"))
    prob_1 = float(row["probabilidade_classe_1"]) if "probabilidade_classe_1" in row and pd.notna(row["probabilidade_classe_1"]) else None
    prob_0 = float(row["probabilidade_classe_0"]) if "probabilidade_classe_0" in row and pd.notna(row["probabilidade_classe_0"]) else None

    label = prediction_label(pred)
    risk_level = classify_risk(prob_1)

    if pred == 1:
        st.error(f"Resultado da predição: **{label}**")
    else:
        st.success(f"Resultado da predição: **{label}**")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Resultado", label)
    with col2:
        st.metric("Probabilidade de risco (classe 1)", f"{prob_1:.2%}" if prob_1 is not None else "-")
    with col3:
        st.metric("Nível estimado", risk_level)

    if prob_1 is not None:
        st.write("### Intensidade do risco estimado")
        st.progress(min(max(prob_1, 0.0), 1.0))
        st.caption(
            "Classe 1 representa risco de defasagem futura. "
            "Classe 0 representa ausência de risco de defasagem futura."
        )

    detail_col1, detail_col2 = st.columns(2)
    with detail_col1:
        if prob_1 is not None:
            st.metric("Prob. classe 1", f"{prob_1:.2%}")
    with detail_col2:
        if prob_0 is not None:
            st.metric("Prob. classe 0", f"{prob_0:.2%}")

    st.write("### Interpretação resumida")
    reasons = build_prediction_explanation(input_df.iloc[0], prob_1)
    for reason in reasons:
        st.write(f"- {reason}")

    with st.expander("Ver dados usados na predição"):
        display_df = format_prediction_output(result_df)
        st.dataframe(display_df, use_container_width=True)

    with st.expander("Entenda o significado do resultado"):
        st.markdown(
            """
**Classe 1**: indica que o modelo estimou risco de o aluno apresentar defasagem no ano seguinte.  
**Classe 0**: indica que o modelo estimou ausência desse risco.

As probabilidades mostram o grau de confiança do modelo em cada uma das classes.
A decisão final considera a classe com maior probabilidade.
            """
        )


def render_batch_prediction():
    st.subheader("Predição em lote")
    st.caption(
        "Envie um CSV, XLSX ou Parquet com as colunas do modelo: "
        "`ano_pede`, `inde`, `n_av`, `iaa`, `ieg`, `ips`, `ipp`, `ida`, "
        "`mat`, `por`, `ing`, `ipv`, `ian`, `fase_ideal`, `defasagem`."
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

    st.write("Prévia do arquivo enviado")
    st.dataframe(batch_df.head(20), use_container_width=True)

    if st.button("Executar predição em lote"):
        try:
            result_df = run_prediction_pipeline(batch_df)
        except Exception as exc:
            st.error(f"Erro ao executar inferência: {exc}")
            return

        display_df = format_prediction_output(result_df)

        st.success(f"Predição concluída para {len(display_df)} registro(s).")

        c1, c2, c3 = st.columns(3)
        c1.metric("Registros processados", len(display_df))
        c2.metric(
            "Com risco",
            int((display_df["predicao"] == 1).sum()) if "predicao" in display_df.columns else 0,
        )
        c3.metric(
            "Sem risco",
            int((display_df["predicao"] == 0).sum()) if "predicao" in display_df.columns else 0,
        )

        if "risco_estimado" in display_df.columns:
            st.write("Distribuição do nível de risco")
            risk_dist = (
                display_df["risco_estimado"]
                .fillna("Indisponível")
                .value_counts()
                .rename_axis("nivel_risco")
                .reset_index(name="quantidade")
            )
            st.dataframe(risk_dist, use_container_width=True)

        st.write("Resultado das predições")
        st.dataframe(display_df.head(100), use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "Baixar CSV",
                data=df_to_download_bytes(display_df, "csv"),
                file_name="predicoes_passos_magicos.csv",
                mime="text/csv",
            )
        with col2:
            st.download_button(
                "Baixar XLSX",
                data=df_to_download_bytes(display_df, "xlsx"),
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
    c3.metric(
        "Anos",
        ", ".join(map(str, sorted(df["ano_pede"].dropna().unique()))) if "ano_pede" in df.columns else "-",
    )

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
    defaults = get_default_values_from_scenario(cfg["scenario"])

    st.title("✨ Passos Mágicos • Solução Preditiva")
    st.caption(
        "Aplicação de Machine Learning para estimar o risco de defasagem escolar futura "
        "a partir dos indicadores educacionais atuais do aluno."
    )

    st.info(
        "Modelo utilizado: Regressão Logística. "
        "Classe 1 = com risco de defasagem futura. Classe 0 = sem risco."
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
        render_single_prediction(defaults)

    with tabs[1]:
        render_batch_prediction()

    with tabs[2]:
        render_data_explorer(cfg["analytic_file"])

    with tabs[3]:
        render_project_status()


if __name__ == "__main__":
    main()