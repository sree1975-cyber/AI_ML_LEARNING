import pandas as pd

def load_data(file_path):
    if file_path.name.endswith('.csv'):
        return pd.read_csv(file_path)
    else:
        return pd.read_excel(file_path)
