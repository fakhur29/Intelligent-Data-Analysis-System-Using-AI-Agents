import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


#######################handle_missing_values########################
def handle_missing_values(df, strategy="mean"):
    """
    Handles missing values using different strategies:
    - mean / median (for numeric)
    - mode (for categorical)
    - drop (removes rows with missing values)
    """
    df_cleaned = df.copy()
    
    columns_with_nulls = df_cleaned.columns[df_cleaned.isnull().any()].tolist()

    for col in columns_with_nulls:
        is_numeric = pd.api.types.is_numeric_dtype(df_cleaned[col])

        if strategy in ["mean", "median"]:
            if is_numeric:
                if strategy == "mean":
                    df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mean())
                else:
                    df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())
            else:
                mode_val = df_cleaned[col].mode()
                if not mode_val.empty:
                    df_cleaned[col] = df_cleaned[col].fillna(mode_val[0])

        elif strategy == "mode":
            mode_val = df_cleaned[col].mode()
            if not mode_val.empty:
                df_cleaned[col] = df_cleaned[col].fillna(mode_val[0])

        elif strategy == "drop":
            df_cleaned = df_cleaned.dropna()
            break

    return df_cleaned


def remove_duplicates(df):
    initial_count = len(df)
    df_cleaned = df.drop_duplicates(keep='first')
    final_count = len(df_cleaned)
    removed_count = initial_count - final_count
    
    print(f"✅ Duplicates Removed: {removed_count}")
    
    return df_cleaned

################################ formet correction#####################################



def format_correction(df):
    """
    Standardizes text and prepares string-based numbers for conversion.
    """
    df_fixed = df.copy()
    
    for col in df_fixed.select_dtypes(include=['object']):
        # 1. Remove leading/trailing whitespace
        df_fixed[col] = df_fixed[col].astype(str).str.strip()
        
        # 2. Standardize Case (Optional: change to Title Case or Lowercase)
        # We'll use Title Case for better readability in reports/graphs
        df_fixed[col] = df_fixed[col].str.lower()
        
        # 3. Remove non-numeric symbols from potential number columns 
        # (Removes $ and , but keeps decimals and numbers)
        if df_fixed[col].str.contains(r'[\$,]', regex=True).any():
            df_fixed[col] = df_fixed[col].replace(r'[\$,]', '', regex=True)
            
    return df_fixed

######################## Fix data types #################################


def fix_data_types(df):
    """
    Automatically converts columns to their most appropriate data types.
    """
    df_fixed = df.copy()
    
    for col in df_fixed.columns:
        # 1. Try converting to numeric (integers or floats)
        # errors='ignore' ensures text like names stay as they are
        converted_numeric = pd.to_numeric(df_fixed[col], errors='coerce')
        if converted_numeric.notna().sum() > 0.7 * len(df_fixed):
            df_fixed[col] = converted_numeric
            
        # 2. Try converting to datetime if the column name suggests time
        if 'date' in col.lower() or 'time' in col.lower():
            try:
                df_fixed[col] = pd.to_datetime(df_fixed[col])
            except:
                pass
        if pd.api.types.is_datetime64_any_dtype(df_fixed[col]):
            df_fixed[col] = df_fixed[col].astype('int64') // 10**9      
    return df_fixed


def handle_outliers(df):
    """
    Handles outliers using IQR method by capping values (not removing rows).
    This prevents data loss.
    """
    df_cleaned = df.copy()
    numeric_cols = df_cleaned.select_dtypes(include=['number']).columns

    for col in numeric_cols:
        Q1 = df_cleaned[col].quantile(0.25)
        Q3 = df_cleaned[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Cap values instead of removing rows
        df_cleaned[col] = df_cleaned[col].clip(lower=lower_bound, upper=upper_bound)

    return df_cleaned

#####################################Encoding cetegorical value###############################

# def encode_categorical(df, threshold=50):
#     """
#     Encodes text columns with unique values below the threshold.
#     Keeps high-cardinality columns (like Names) in their original text format.
#     """
#     df_encoded = df.copy()
#     le = LabelEncoder()
    
#     # Select text columns (object or category)
#     categorical_cols = df_encoded.select_dtypes(include=['object', 'category']).columns
    
#     for col in categorical_cols:
#         num_unique = df_encoded[col].nunique()
        
#         if num_unique == 2:
#             df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
#         elif num_unique <= threshold:
#             df_encoded = pd.get_dummies(df_encoded, columns=[col])
#         else:
#             # Just skip the encoding and keep the column as it is
#             print(f"Skipping encoding for '{col}' ({num_unique} unique values). Keeping as text.")
                
#     return df_encoded


#########################scaling features##########################################


# def scale_features(df):
#     """
#     Standardizes continuous numerical columns (Mean=0, StdDev=1).
#     Avoids scaling binary/encoded columns (e.g., 0/1).
#     """
#     df_scaled = df.copy()
#     scaler = StandardScaler()
    
#     # Select only continuous numeric columns (exclude binary/encoded)
#     numeric_cols = [
#         col for col in df_scaled.select_dtypes(include=['number']).columns
#         if df_scaled[col].nunique() > 2
#     ]
    
#     if len(numeric_cols) > 0:
#         df_scaled[numeric_cols] = scaler.fit_transform(df_scaled[numeric_cols])
        
#     return df_scaled

#########################removing irrelevant features##########################################

def drop_irrelevant_columns(df):
    """
    Automatically detects and drops columns that are likely IDs, Names, 
    or contain no useful information (zero variance).
    """
    df_clean = df.copy()
    
    # 1. List of keywords that usually indicate useless columns for AI
    irrelevant_keywords = ['id', 'name', 'serial', 'index', 'timestamp', 'rollno', 'ssn', 'ticket', 'passengerid']    
    cols_to_drop = []
    
    for col in df_clean.columns:
        # Check if column name contains any irrelevant keyword (case-insensitive)
        if any(keyword in col.lower() for keyword in irrelevant_keywords):
            cols_to_drop.append(col)
            continue
            
        # 2. Check for "Zero Variance" (all values are the same)
        if df_clean[col].nunique() <= 1:
            cols_to_drop.append(col)
            continue
            
        
        # if df_clean[col].dtype == 'object' and df_clean[col].nunique() == len(df_clean) and len(df_clean) > 50:
        #     cols_to_drop.append(col)
        # Drop only if extremely high cardinality (almost unique)
        if df_clean[col].dtype == 'object':
            unique_ratio = df_clean[col].nunique() / len(df_clean)
            if unique_ratio > 0.75:   # safer threshold
                cols_to_drop.append(col)

    # Drop the identified columns
    df_clean = df_clean.drop(columns=cols_to_drop)
    
    print(f"Dropped columns: {cols_to_drop}")
    return df_clean