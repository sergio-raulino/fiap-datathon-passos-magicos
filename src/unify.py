from __future__ import annotations

from pathlib import Path
from typing import Dict
import re

import pandas as pd


RAW_FILE = "BASE DE DADOS PEDE 2024 - DATATHON.xlsx"


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    col_map = {
        "RA": "ra",
        "Fase": "fase",
        "Turma": "turma",
        "Nome": "nome",
        "Nome Anonimizado": "nome",
        "Ano nasc": "ano_nasc",
        "Data de Nasc": "data_nasc",
        "Idade 22": "idade",
        "Idade": "idade",
        "Gênero": "genero",
        "Ano ingresso": "ano_ingresso",
        "Instituição de ensino": "instituicao_ensino",
        "Escola": "escola",
        "Pedra 20": "pedra_2020",
        "Pedra 21": "pedra_2021",
        "Pedra 22": "pedra_2022",
        "Pedra 2023": "pedra_2023",
        "Pedra 2024": "pedra_2024",
        "INDE 22": "inde_2022",
        "INDE 2023": "inde_2023",
        "INDE 2024": "inde_2024",
        "Cg": "cg",
        "Cf": "cf",
        "Ct": "ct",
        "Nº Av": "n_av",
        "Avaliador1": "avaliador_1",
        "Avaliador2": "avaliador_2",
        "Avaliador3": "avaliador_3",
        "Avaliador4": "avaliador_4",
        "Avaliador5": "avaliador_5",
        "Avaliador6": "avaliador_6",
        "Rec Av1": "rec_av1",
        "Rec Av2": "rec_av2",
        "Rec Av3": "rec_av3",
        "Rec Av4": "rec_av4",
        "Rec Av5": "rec_av5",
        "Rec Av6": "rec_av6",
        "IAA": "iaa",
        "IEG": "ieg",
        "IPS": "ips",
        "IPP": "ipp",
        "Rec Psicologia": "rec_psicologia",
        "IDA": "ida",
        "Matem": "mat",
        "Portug": "por",
        "Inglês": "ing",
        "Indicado": "indicado",
        "Atingiu PV": "atingiu_pv",
        "IPV": "ipv",
        "IAN": "ian",
        "Fase ideal": "fase_ideal",
        "Defas": "defasagem",
        "Defasagem": "defasagem",
        "Destaque IEG": "destaque_ieg",
        "Destaque IDA": "destaque_ida",
        "Destaque IPV": "destaque_ipv",
        "Destaque IVP": "destaque_ipv",
        "Ativo/Inativo": "ativo_inativo",
    }
    return df.rename(columns=col_map)


def normalize_fase(value, year: int) -> pd.NA | str:
    if pd.isna(value):
        return pd.NA

    s = str(value).strip().upper()

    if year == 2022:
        if s.isdigit():
            return str(int(s))
        return s

    if year == 2023:
        if s == "ALFA":
            return "0"

        m = re.fullmatch(r"FASE\s+(\d+)", s)
        if m:
            return str(int(m.group(1)))

        if s.isdigit():
            return str(int(s))

        return s

    if year == 2024:
        if s == "ALFA":
            return "0"

        # Ex.: 1A, 2B, 8R -> 1, 2, 8
        m = re.fullmatch(r"([0-9]+)[A-Z]+", s)
        if m:
            return str(int(m.group(1)))

        # número puro: mantém, inclusive 9
        if s.isdigit():
            return str(int(s))

        return pd.NA

    if s.isdigit():
        return str(int(s))
    return s


def normalize_turma(value, year: int) -> pd.NA | str:
    if pd.isna(value):
        return pd.NA

    s = str(value).strip().upper()

    if year == 2022:
        if re.fullmatch(r"[A-Z]+", s):
            return s
        return s

    if year in (2023, 2024):
        # Ex.: "ALFA A - G0/G1" -> "A"
        m = re.match(r"^ALFA\s+([A-Z])\b", s)
        if m:
            return m.group(1)

        # Ex.: "8A", "2R" -> "A", "R"
        m = re.fullmatch(r"[0-9]+([A-Z]+)", s)
        if m:
            return m.group(1)

        # Já é letra
        if re.fullmatch(r"[A-Z]+", s):
            return s

        # Numérico puro, como "9" -> NA
        if s.isdigit():
            return pd.NA

        return pd.NA

    return s


def standardize_sheet(df: pd.DataFrame, year: int) -> pd.DataFrame:
    df = normalize_columns(df).copy()
    df["ano_pede"] = year

    desired_columns = [
        "ano_pede",
        "ra",
        "fase",
        "turma",
        "nome",
        "ano_nasc",
        "data_nasc",
        "idade",
        "genero",
        "ano_ingresso",
        "instituicao_ensino",
        "escola",
        "ativo_inativo",
        "pedra_2020",
        "pedra_2021",
        "pedra_2022",
        "pedra_2023",
        "pedra_2024",
        "inde_2022",
        "inde_2023",
        "inde_2024",
        "cg",
        "cf",
        "ct",
        "n_av",
        "avaliador_1",
        "rec_av1",
        "avaliador_2",
        "rec_av2",
        "avaliador_3",
        "rec_av3",
        "avaliador_4",
        "rec_av4",
        "avaliador_5",
        "rec_av5",
        "avaliador_6",
        "rec_av6",
        "iaa",
        "ieg",
        "ips",
        "ipp",
        "rec_psicologia",
        "ida",
        "mat",
        "por",
        "ing",
        "indicado",
        "atingiu_pv",
        "ipv",
        "ian",
        "fase_ideal",
        "defasagem",
        "destaque_ieg",
        "destaque_ida",
        "destaque_ipv",
    ]

    for col in desired_columns:
        if col not in df.columns:
            df[col] = pd.NA

    df = df[desired_columns]

    df["fase"] = df["fase"].apply(lambda x: normalize_fase(x, year))
    df["turma"] = df["turma"].apply(lambda x: normalize_turma(x, year))

    return df


def read_raw_sheets(raw_path: str | Path) -> Dict[str, pd.DataFrame]:
    xls = pd.ExcelFile(raw_path)
    return {
        "PEDE2022": pd.read_excel(xls, sheet_name="PEDE2022"),
        "PEDE2023": pd.read_excel(xls, sheet_name="PEDE2023"),
        "PEDE2024": pd.read_excel(xls, sheet_name="PEDE2024"),
    }


def build_analytic_base(raw_path: str | Path) -> pd.DataFrame:
    sheets = read_raw_sheets(raw_path)

    df_2022 = standardize_sheet(sheets["PEDE2022"], 2022)
    df_2023 = standardize_sheet(sheets["PEDE2023"], 2023)
    df_2024 = standardize_sheet(sheets["PEDE2024"], 2024)

    df = pd.concat([df_2022, df_2023, df_2024], ignore_index=True)

    if "data_nasc" in df.columns:
        df["data_nasc"] = pd.to_datetime(df["data_nasc"], errors="coerce")

    numeric_cols = [
        "idade", "ano_ingresso",
        "inde_2022", "inde_2023", "inde_2024",
        "cg", "cf", "ct", "n_av",
        "iaa", "ieg", "ips", "ipp", "ida",
        "mat", "por", "ing",
        "ipv", "ian", "defasagem",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    text_cols = [
        "ra", "fase", "turma", "nome", "genero", "instituicao_ensino",
        "escola", "ativo_inativo", "fase_ideal",
        "pedra_2020", "pedra_2021", "pedra_2022", "pedra_2023", "pedra_2024",
        "indicado", "atingiu_pv",
        "destaque_ieg", "destaque_ida", "destaque_ipv",
    ]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip()

    return df


def export_bases(raw_path: str | Path, output_dir: str | Path) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = build_analytic_base(raw_path)

    analytic_path = output_dir / "Base de Dados PEDE.xlsx"
    parquet_path = output_dir / "base_analitica.parquet"

    df.to_excel(analytic_path, index=False)
    df.to_parquet(parquet_path, index=False)

    return analytic_path, parquet_path