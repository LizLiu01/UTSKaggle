import pandas as pd

def categorise(df):
    for c in df.columns:
        if df[c].dtype == 'object':
            df[c] = pd.Categorical(df[c]).codes