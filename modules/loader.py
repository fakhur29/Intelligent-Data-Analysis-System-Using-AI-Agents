import pandas as pd
import os

SUPPORTED_EXTENSIONS = [".csv", ".xlsx", ".xls", ".json"]

def load_file(file_path: str):
    ext = os.path.splitext(file_path)[1].lower()

    if ext not in SUPPORTED_EXTENSIONS:
        return {
            "success": False,
            "error": f"Unsupported file type '{ext}'. Supported: CSV, Excel, JSON",
            "df": None
        }

    if not os.path.exists(file_path):
        return {
            "success": False,
            "error": "File not found",
            "df": None
        }

    try:
        if ext == ".csv":
            
            df = pd.read_csv(file_path, low_memory=False,encoding='utf-8', errors='replace')

        elif ext in [".xlsx", ".xls"]:
            df = pd.read_excel(file_path)

        elif ext == ".json":
            df = pd.read_json(file_path, orient="records")

        return {
            "success": True,
            "df": df,
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": df.dtypes.astype(str).to_dict()
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "df": None
        }
