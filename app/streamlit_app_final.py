from __future__ import annotations

import io
import pickle
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

DEFAULT_ANALYTIC_FILE = ROOT_DIR / "data" / "processed" / "base_PEDE_consolidada_analitica.parquet"
ARTIFACTS_DIR = ROOT_DIR / "data" / "artifacts"

MODEL_FEATURES = [
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


def risk_level_from_prob_risk(prob_risk: float | None) -> str:
    if prob_risk is None:
        return "Indisponível"
    if prob_risk < 0.30:
        return "Baixo"
    if prob_risk < 0.70:
        return "Moderado"
    return "Alto"


def prediction_label(pred: Any) -> str:
    """
    Conforme seu teste:
    - classe 0 = alto risco
    - classe 1 = baixo risco
    """
    if pred in [0, "0", False]:
        return "Com risco de defasagem futura"
    return "Sem risco de defasagem futura"


def build_prediction_explanation(input_row: pd.Series, prob_risk: float | None) -> list[str]:
    reasons: list[str] = []

    def val(col: str, default: float = 0.0) -> float:
        x = input_row.get(col, default)
        try:
            return float(x)
        except Exception:
            return default

    fase = str(input_row.get("fase_ideal", "")).strip()

    # sinais favoráveis
    if val("ian") >= 7:
        reasons.append("O IAN está elevado, o que neste perfil tende a favorecer um cenário de menor risco.")
    if val("n_av") >= 6:
        reasons.append("O número de avaliações está alto, indicando padrão mais próximo de baixo risco.")
    if val("ipv") >= 7:
        reasons.append("O IPV está alto, contribuindo para uma previsão mais favorável.")
    if val("ieg") >= 7:
        reasons.append("O IEG elevado reforça um perfil associado a menor risco.")
    if val("ipp") >= 7:
        reasons.append("O IPP alto também aproxima o registro de um cenário de baixo risco.")
    if "Fase 7" in fase:
        reasons.append("A fase ideal informada está compatível com um cenário que, neste modelo, tende a menor risco.")

    # sinais de risco
    if val("ian") <= 4:
        reasons.append("O IAN baixo é um sinal que aproxima a previsão de maior risco.")
    if val("n_av") <= 3:
        reasons.append("Poucas avaliações informadas tendem a puxar a predição para maior risco.")
    if val("ipv") <= 4:
        reasons.append("O IPV baixo reforça a possibilidade de maior risco.")
    if val("ieg") <= 4:
        reasons.append("O IEG baixo também é compatível com um cenário mais crítico.")
    if val("ipp") <= 4:
        reasons.append("O IPP baixo contribui para maior probabilidade de risco.")

    if prob_risk is not None:
        level = risk_level_from_prob_risk(prob_risk).lower()
        reasons.append(f"A probabilidade estimada foi classificada como risco {level}.")

    if not reasons:
        reasons.append("A previsão foi calculada com base nos indicadores informados e no padrão histórico aprendido pelo modelo.")

    return reasons


def format_prediction_output(result_df: pd.DataFrame) -> pd.DataFrame:
    output = result_df.copy()

    if "predicao" in output.columns:
        output["resultado_predicao"] = output["predicao"].apply(prediction_label)

    if "probabilidade_classe_0" in output.columns:
        output["probabilidade_alto_risco"] = output["probabilidade_classe_0"]
        output["risco_estimado"] = output["probabilidade_classe_0"].apply(risk_level_from_prob_risk)
        output["probabilidade_classe_0_pct"] = output["probabilidade_classe_0"].apply(lambda x: f"{float(x):.2%}")

    if "probabilidade_classe_1" in output.columns:
        output["probabilidade_baixo_risco"] = output["probabilidade_classe_1"]
        output["probabilidade_classe_1_pct"] = output["probabilidade_classe_1"].apply(lambda x: f"{float(x):.2%}")

    return output


def find_first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


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
# Fallback local de artefatos
# =========================================================
def load_local_artifacts_fallback() -> dict[str, Any]:
    """
    Fallback para quando src/model.py não estiver importável.
    Tenta carregar diretamente de data/artifacts.
    """
    candidate_files = [
        ARTIFACTS_DIR / "model.pkl",
        ARTIFACTS_DIR / "pipeline.pkl",
        ARTIFACTS_DIR / "artifacts.pkl",
        ARTIFACTS_DIR / "model.joblib",
        ARTIFACTS_DIR / "pipeline.joblib",
        ARTIFACTS_DIR / "artifacts.joblib",
    ]

    chosen = find_first_existing(candidate_files)
    if chosen is None:
        raise FileNotFoundError(
            "Não encontrei artefatos do modelo em data/artifacts. "
            "Esperado algo como model.pkl, pipeline.pkl ou artifacts.pkl."
        )

    if chosen.suffix == ".pkl":
        with open(chosen, "rb") as f:
            obj = pickle.load(f)
    else:
        try:
            import joblib
        except Exception as exc:
            raise RuntimeError(
                "Foi encontrado um arquivo .joblib, mas a biblioteca joblib não está disponível."
            ) from exc
        obj = joblib.load(chosen)

    if isinstance(obj, dict):
        if "model" in obj:
            return obj
        return {"model": obj, "raw_artifact": obj}

    return {"model": obj, "raw_artifact": obj}


# =========================================================
# Cached loading
# =========================================================
@st.cache_resource(show_spinner=False)
def load_model_resource():
    if load_model_fn is not None:
        try:
            return load_model_fn()
        except Exception:
            pass

    return load_local_artifacts_fallback()


@st.cache_data(show_spinner=False)
def load_analytic_base(file_path: str) -> pd.DataFrame:
    path = Path(file_path)

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


def ensure_model_features(input_df: pd.DataFrame) -> pd.DataFrame:
    missing = [col for col in MODEL_FEATURES if col not in input_df.columns]
    if missing:
        raise ValueError(
            "O arquivo não contém todas as colunas esperadas pelo modelo. "
            f"Faltando: {missing}"
        )
    return input_df[MODEL_FEATURES].copy()


def apply_feature_pipeline(input_df: pd.DataFrame, model_bundle: Any) -> pd.DataFrame:
    """
    Se existir src/features.py, usa a função encontrada.
    Caso contrário, envia as colunas diretamente ao pipeline/modelo.
    """
    if prepare_inference_features_fn is None:
        return ensure_model_features(input_df)

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
        raise RuntimeError("Não foi possível carregar o modelo.")

    if clean_dataframe_fn is not None:
        try:
            input_df = clean_dataframe_fn(input_df)
        except Exception:
            pass

    X = apply_feature_pipeline(input_df.copy(), model_bundle)
    model_obj = model_bundle_to_model(model_bundle)

    if model_obj is None:
        raise RuntimeError("Modelo não carregado corretamente.")

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
        classes_ = None

        try:
            if hasattr(model_obj, "classes_"):
                classes_ = list(model_obj.classes_)
            elif hasattr(model_obj, "named_steps") and "model" in model_obj.named_steps:
                inner_model = model_obj.named_steps["model"]
                if hasattr(inner_model, "classes_"):
                    classes_ = list(inner_model.classes_)
        except Exception:
            classes_ = None

        if classes_ is not None and len(classes_) == proba_df.shape[1]:
            for idx, cls in enumerate(classes_):
                result[f"probabilidade_classe_{cls}"] = proba_df.iloc[:, idx]
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
        value=str(DEFAULT_ANALYTIC_FILE),
    )

    st.sidebar.markdown("---")
    st.sidebar.caption("A aplicação tenta usar `src/` quando existir, mas também funciona com fallback direto em `data/artifacts/`.")

    st.sidebar.markdown("### Cenários rápidos")
    scenario = st.sidebar.selectbox(
        "Selecione um exemplo para preencher na predição individual",
        options=[
            "Nenhum",
            "Cenário de alto risco",
            "Cenário de baixo risco",
        ],
        index=0,
    )

    return {
        "analytic_file": analytic_file,
        "scenario": scenario,
    }


def get_default_values_from_scenario(scenario: str) -> dict[str, Any]:
    base = {
        "inde": 6.0,
        "n_av": 4.0,
        "iaa": 6.0,
        "ieg": 6.0,
        "ips": 6.0,
        "ipp": 6.0,
        "ida": 6.0,
        "mat": 6.0,
        "por": 6.0,
        "ing": 6.0,
        "ipv": 6.0,
        "ian": 6.0,
        "fase_ideal": "Fase 4 (9º ano)",
    }

    if scenario == "Cenário de alto risco":
        return {
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
            "fase_ideal": "Fase 4 (9º ano)",
        }

    if scenario == "Cenário de baixo risco":
        return {
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
            "fase_ideal": "Fase 7 (3º EM)",
        }

    return base


def render_project_status():
    st.subheader("Status técnico")

    rows = [
        {
            "módulo": "src/features.py",
            "função": prepare_inference_features_fn.__name__ if prepare_inference_features_fn else "fallback direto para features do modelo",
            "status": "OK" if prepare_inference_features_fn else "fallback",
        },
        {
            "módulo": "src/model.py (load)",
            "função": load_model_fn.__name__ if load_model_fn else "fallback em data/artifacts",
            "status": "OK" if load_model_fn else "fallback",
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

    try:
        model_bundle = load_model_resource()
        model_obj = model_bundle_to_model(model_bundle)
        classes_ = None

        if hasattr(model_obj, "classes_"):
            classes_ = list(model_obj.classes_)
        elif hasattr(model_obj, "named_steps") and "model" in model_obj.named_steps:
            inner_model = model_obj.named_steps["model"]
            if hasattr(inner_model, "classes_"):
                classes_ = list(inner_model.classes_)

        if classes_ is not None:
            st.caption(f"Classes carregadas do modelo: {classes_}")
            st.caption("Interpretação usada na interface: classe 0 = alto risco | classe 1 = baixo risco")
    except Exception as exc:
        st.warning(f"Não foi possível inspecionar as classes do modelo: {exc}")


def render_single_prediction(defaults: dict[str, Any]):
    st.subheader("Predição individual")
    st.caption(
        "Informe os indicadores atuais do aluno para estimar o risco. "
        "Nesta interface: classe 0 = alto risco e classe 1 = baixo risco."
    )

    with st.form("predicao_individual"):
        col1, col2, col3 = st.columns(3)

        fase_options = [
            "ALFA  (2º e 3º ano)",
            "Fase 1 (4º ano)",
            "Fase 2 (5º e 6º ano)",
            "Fase 3 (7º e 8º ano)",
            "Fase 4 (9º ano)",
            "Fase 5 (1º EM)",
            "Fase 6 (2º EM)",
            "Fase 7 (3º EM)",
            "Fase 8 (Universitários)",
        ]
        fase_default_index = fase_options.index(defaults["fase_ideal"]) if defaults["fase_ideal"] in fase_options else 4

        with col1:
            inde = st.number_input("INDE", min_value=0.0, max_value=10.0, value=float(defaults["inde"]), step=0.1)
            n_av = st.number_input("Nº de avaliações", min_value=0.0, max_value=20.0, value=float(defaults["n_av"]), step=1.0)
            iaa = st.number_input("IAA", min_value=0.0, max_value=10.0, value=float(defaults["iaa"]), step=0.1)
            ieg = st.number_input("IEG", min_value=0.0, max_value=10.0, value=float(defaults["ieg"]), step=0.1)
            ips = st.number_input("IPS", min_value=0.0, max_value=10.0, value=float(defaults["ips"]), step=0.1)

        with col2:
            ipp = st.number_input("IPP", min_value=0.0, max_value=10.0, value=float(defaults["ipp"]), step=0.1)
            ida = st.number_input("IDA", min_value=0.0, max_value=10.0, value=float(defaults["ida"]), step=0.1)
            mat = st.number_input("Matemática", min_value=0.0, max_value=10.0, value=float(defaults["mat"]), step=0.1)
            por = st.number_input("Português", min_value=0.0, max_value=10.0, value=float(defaults["por"]), step=0.1)
            ing = st.number_input("Inglês", min_value=0.0, max_value=10.0, value=float(defaults["ing"]), step=0.1)

        with col3:
            ipv = st.number_input("IPV", min_value=0.0, max_value=10.0, value=float(defaults["ipv"]), step=0.1)
            ian = st.number_input("IAN", min_value=0.0, max_value=10.0, value=float(defaults["ian"]), step=0.1)
            fase_ideal = st.selectbox("Fase ideal", options=fase_options, index=fase_default_index)

        submitted = st.form_submit_button("Gerar predição")

    if not submitted:
        return

    input_df = pd.DataFrame(
        [
            {
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
    prob_0 = float(row["probabilidade_classe_0"]) if "probabilidade_classe_0" in row and pd.notna(row["probabilidade_classe_0"]) else None
    prob_1 = float(row["probabilidade_classe_1"]) if "probabilidade_classe_1" in row and pd.notna(row["probabilidade_classe_1"]) else None

    label = prediction_label(pred)
    risk_level = risk_level_from_prob_risk(prob_0)

    if pred == 0:
        st.error(f"Resultado da predição: **{label}**")
    else:
        st.success(f"Resultado da predição: **{label}**")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Resultado", label)
    with c2:
        st.metric("Probabilidade de alto risco", f"{prob_0:.2%}" if prob_0 is not None else "-")
    with c3:
        st.metric("Nível estimado de risco", risk_level)

    if prob_0 is not None:
        st.write("### Intensidade do risco estimado")
        st.progress(min(max(prob_0, 0.0), 1.0))
        st.caption(
            "Interpretação da interface: "
            "classe 0 = alto risco | classe 1 = baixo risco."
        )

    d1, d2 = st.columns(2)
    with d1:
        if prob_0 is not None:
            st.metric("Prob. classe 0 (alto risco)", f"{prob_0:.2%}")
    with d2:
        if prob_1 is not None:
            st.metric("Prob. classe 1 (baixo risco)", f"{prob_1:.2%}")

    st.write("### Interpretação resumida")
    reasons = build_prediction_explanation(input_df.iloc[0], prob_0)
    for reason in reasons:
        st.write(f"- {reason}")

    with st.expander("Ver dados usados na predição"):
        display_df = format_prediction_output(result_df)
        st.dataframe(display_df, use_container_width=True)

    with st.expander("Entenda o significado do resultado"):
        st.markdown(
            """
**Classe 0**: maior risco.  
**Classe 1**: menor risco.

As probabilidades mostram o grau de confiança do modelo em cada classe.  
A decisão final considera a classe com maior probabilidade.
            """
        )


def render_batch_prediction():
    st.subheader("Predição em lote")
    st.caption(
        "Envie um CSV, XLSX ou Parquet com as colunas do modelo: "
        "`inde`, `n_av`, `iaa`, `ieg`, `ips`, `ipp`, `ida`, "
        "`mat`, `por`, `ing`, `ipv`, `ian`, `fase_ideal`."
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
            "Alto risco (classe 0)",
            int((display_df["predicao"] == 0).sum()) if "predicao" in display_df.columns else 0,
        )
        c3.metric(
            "Baixo risco (classe 1)",
            int((display_df["predicao"] == 1).sum()) if "predicao" in display_df.columns else 0,
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
        "Aplicação de Machine Learning para inferência preditiva com base nos "
        "indicadores educacionais do aluno."
    )

    st.info(
        "Interpretação adotada nesta versão: "
        "classe 0 = alto risco | classe 1 = baixo risco."
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