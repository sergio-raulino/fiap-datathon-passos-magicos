import pandas as pd


def load_data(path):
    return pd.read_excel(path)


def clean_data(df):

    df.columns = df.columns.str.strip()

    df = df.drop_duplicates()

    df = df.replace("-", None)

    return df