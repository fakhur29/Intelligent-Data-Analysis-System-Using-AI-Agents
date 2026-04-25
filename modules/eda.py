import pandas as pd
import numpy as np

########################basic info ############################

def get_basic_info(df):
    """
    EDA Task 1: Basic Data Overview.
    Returns the shape, column names, and data types.
    """
    rows, cols = df.shape
    
    result = {
        "total_rows": rows,
        "total_columns": cols,
        "column_names": list(df.columns),
        "data_types": df.dtypes.astype(str).to_dict()
    }
    
    print("--- 1. BASIC DATA OVERVIEW ---")
    print(f"Total Rows: {rows}")
    print(f"Total Columns: {cols}")
    print("\nColumn Details & Data Types:")
    print(df.dtypes)
    print("-" * 30)
    
    return result


###########################get_statistical_summary(df) ############################

def get_statistical_summary(df):
    """
    EDA Task 2: Statistical Summary.
    Returns Mean, Median, Min, Max, and Std Dev for numeric columns.
    """
    numeric_df = df.select_dtypes(include=['number'])
    
    if not numeric_df.empty:
        summary = numeric_df.describe().T
        summary = summary[['mean', '50%', 'std', 'min', 'max']]
        summary.columns = ['Mean', 'Median', 'Std Dev', 'Min', 'Max']
        
        result = {
            "success": True,
            "summary": summary.round(2).to_dict('index')
        }
        
        print("\n--- 2. STATISTICAL SUMMARY ---")
        print(summary.round(2))
    else:
        result = {
            "success": False,
            "summary": {},
            "message": "No numeric columns found for statistical summary."
        }
        print("\n--- 2. STATISTICAL SUMMARY ---")
        print("No numeric columns found for statistical summary.")
    
    print("-" * 30)
    return result


################################missing value analysis##########################################

def get_missing_values_analysis(df):
    """
    EDA Task 3: Missing Values Analysis.
    Returns count and percentage of missing values for each column.
    """
    missing_count = df.isnull().sum()
    total_rows = len(df)
    missing_percentage = (missing_count / total_rows) * 100
    
    missing_df = pd.DataFrame({
        'Missing Values': missing_count,
        'Percentage (%)': missing_percentage.round(2)
    })
    
    only_missing = missing_df[missing_df['Missing Values'] > 0]
    
    if not only_missing.empty:
        result = {
            "success": True,
            "has_missing": True,            
            "missing_data": only_missing.reset_index().to_dict('records')
        }
        print("\n--- 3. MISSING VALUES ANALYSIS ---")
        print(only_missing)
    else:
        result = {
            "success": True,
            "has_missing": False,
            "missing_data": {},
            "message": "No missing values detected in any column. Perfect!"
        }
        print("\n--- 3. MISSING VALUES ANALYSIS ---")
        print("No missing values detected in any column. Perfect!")
    
    print("-" * 30)
    return result


###########################unique value analysis##########################################

def get_unique_values_analysis(df):
    """
    EDA Task 4: Unique Values Analysis.
    Returns unique values per column to identify categories vs. IDs.
    """
    unique_counts = []
    
    for col in df.columns:
        count = df[col].nunique()
        dtype = str(df[col].dtype)
        
        if count == 1:
            category = "Constant (Zero Variance)"
        elif count == 2:
            category = "Binary"
        elif count < 0.05 * len(df):
            category = "Categorical (Low)"
        elif count == len(df):
            category = "Unique ID / High Cardinality"
        else:
            category = "Continuous / Multi-category"
            
        unique_counts.append({
            'Column': col,
            'Unique Count': count,
            'Data Type': dtype,
            'Category Type': category
        })
    
    unique_df = pd.DataFrame(unique_counts)
    
    result = {
        "success": True,
        "unique_analysis": unique_df.to_dict('records')
    }
    
    print("\n--- 4. UNIQUE VALUES ANALYSIS ---")
    print(unique_df.to_string(index=False))
    print("-" * 30)
    
    return result


##############################correlation analysis##########################################

def get_correlation_analysis(df):
    """
    EDA Task 5: Correlation Analysis.
    Returns the Pearson correlation matrix for numeric columns.
    """
    numeric_df = df.select_dtypes(include=['number'])
    
    if numeric_df.shape[1] < 2:
        result = {
            "success": False,
            "correlation_matrix": {},
            "strong_relationships": [],
            "message": "Not enough numeric columns to perform correlation analysis."
        }
        print("\n--- 5. CORRELATION ANALYSIS ---")
        print("Not enough numeric columns to perform correlation analysis.")
    else:
        corr_matrix = numeric_df.corr()
        
        strong_relationships = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if abs(corr_matrix.iloc[i, j]) > 0.7:
                    col1 = corr_matrix.columns[i]
                    col2 = corr_matrix.columns[j]
                    val = corr_matrix.iloc[i, j]
                    strong_relationships.append({
                        'Column 1': col1,
                        'Column 2': col2,
                        'Correlation': round(val, 2)
                    })
        
        result = {
            "success": True,
            "correlation_matrix": corr_matrix.round(2).to_dict(),
            "strong_relationships": strong_relationships
        }
        
        print("\n--- 5. CORRELATION ANALYSIS ---")
        print("Correlation Matrix:")
        print(corr_matrix.round(2))
        
        if strong_relationships:
            print("\nStrong Relationships detected (|r| > 0.7):")
            for rel in strong_relationships:
                print(f"- {rel['Column 1']} and {rel['Column 2']}: {rel['Correlation']}")
        else:
            print("\nNo strong relationships detected (|r| > 0.7)")
    
    print("-" * 30)
    return result

###################################distrubution analysis##########################################

def get_distribution_analysis(df):
    """
    EDA Task 6: Distribution Analysis.
    Returns Skewness for numeric columns to identify data balance.
    """
    numeric_df = df.select_dtypes(include=['number'])
    
    if not numeric_df.empty:
        distribution_data = []
        
        for col in numeric_df.columns:
            skew_val = numeric_df[col].skew()
            
            if abs(skew_val) < 0.5:
                shape = "Fairly Symmetrical (Normal)"
            elif skew_val > 0.5:
                shape = "Right Skewed (Long tail on right)"
            else:
                shape = "Left Skewed (Long tail on left)"
            
            distribution_data.append({
                'Column': col,
                'Skewness': round(skew_val, 2),
                'Shape': shape
            })
        
        result = {
            "success": True,
            "distribution_analysis": distribution_data
        }
        
        print("\n--- 6. DISTRIBUTION ANALYSIS ---")
        print(f"{'Column':<20} | {'Skewness':<10} | {'Shape'}")
        print("-" * 45)
        for item in distribution_data:
            print(f"{item['Column']:<20} | {item['Skewness']:<10} | {item['Shape']}")
    else:
        result = {
            "success": False,
            "distribution_analysis": [],
            "message": "No numeric columns available to check distribution."
        }
        print("\n--- 6. DISTRIBUTION ANALYSIS ---")
        print("No numeric columns available to check distribution.")
    
    print("-" * 30)
    return result


#################################outlier analysis##########################################

def get_outlier_analysis(df):
    """
    EDA Task 7: Outlier Detection (EDA view).
    Returns the count of outliers in numeric columns using the IQR method.
    """
    numeric_df = df.select_dtypes(include=['number'])
    
    if not numeric_df.empty:
        outlier_data = []
        
        for col in numeric_df.columns:
            Q1 = numeric_df[col].quantile(0.25)
            Q3 = numeric_df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = numeric_df[(numeric_df[col] < lower_bound) | (numeric_df[col] > upper_bound)]
            count = len(outliers)
            
            outlier_data.append({
                'Column': col,
                'Outlier Count': count,
                'Lower Bound': round(lower_bound, 2),
                'Upper Bound': round(upper_bound, 2)
            })
        
        result = {
            "success": True,
            "outlier_analysis": outlier_data
        }
        
        print("\n--- 7. OUTLIER DETECTION (EDA VIEW) ---")
        outlier_df = pd.DataFrame(outlier_data)
        print(outlier_df.to_string(index=False))
    else:
        result = {
            "success": False,
            "outlier_analysis": [],
            "message": "No numeric columns available for outlier detection."
        }
        print("\n--- 7. OUTLIER DETECTION (EDA VIEW) ---")
        print("No numeric columns available for outlier detection.")
    
    print("-" * 30)
    return result

############################get_target_analysis############################

def get_target_analysis(df, target_column):
    """
    EDA Task 8: Target Variable Analysis.
    Returns the distribution analysis of the target column.
    """
    print(f"\n--- 8. TARGET VARIABLE ANALYSIS ({target_column}) ---")
    
    if target_column not in df.columns:
        result = {
            "success": False,
            "error": f"Column '{target_column}' not found in dataset.",
            "target_type": None,
            "target_analysis": {}
        }
        print(f"Error: Column '{target_column}' not found in dataset.")
        print("-" * 30)
        return result

    target = df[target_column]
    
    if target.dtype == 'object' or target.nunique() < 10:
        # Classification
        counts = target.value_counts()
        percent = target.value_counts(normalize=True) * 100
        
        analysis_dict = {
            str(idx): {'Count': int(val), 'Percentage (%)': round(percent[idx], 2)}
            for idx, val in counts.items()
        }
        
        result = {
            "success": True,
            "target_type": "Classification",
            "target_analysis": analysis_dict
        }
        
        print("Type: Classification Target")
        analysis = pd.DataFrame({
            'Count': counts,
            'Percentage (%)': percent.round(2)
        })
        print(analysis)
    else:
        # Regression
        result = {
            "success": True,
            "target_type": "Regression",
            "target_analysis": {
                'Min': float(target.min()),
                'Max': float(target.max()),
                'Average': round(float(target.mean()), 2),
                'Std Dev': round(float(target.std()), 2)
            }
        }
        
        print("Type: Regression Target")
        print(f"Min: {target.min()} | Max: {target.max()}")
        print(f"Average Value: {target.mean():.2f}")
        print(f"Standard Deviation: {target.std():.2f}")

    print("-" * 30)
    return result

################################3get_automated_insights############################

def get_automated_insights(df):
    """
    EDA Task 9: Automated Insights.
    Returns a summary of findings and recommendations.
    """
    insights = []
    
    # 1. Missing Value Insight
    missing_pct = (df.isnull().sum().sum() / df.size) * 100
    if missing_pct > 0:
        insights.append(f"ACTION: Your data has {missing_pct:.2f}% missing values. Run Data Cleaning.")
    
    # 2. Outlier Insight
    numeric_df = df.select_dtypes(include=['number'])
    outlier_found = False
    for col in numeric_df.columns:
        Q1, Q3 = numeric_df[col].quantile(0.25), numeric_df[col].quantile(0.75)
        IQR = Q3 - Q1
        if ((numeric_df[col] < (Q1 - 1.5 * IQR)) | (numeric_df[col] > (Q3 + 1.5 * IQR))).any():
            outlier_found = True
            break
    if outlier_found:
        insights.append("WARNING: Outliers detected. Consider Scaling or Capping.")

    # 3. Skewness Insight
    high_skew = [col for col in numeric_df.columns if abs(numeric_df[col].skew()) > 1]
    if high_skew:
        insights.append(f"NOTE: {len(high_skew)} columns are highly skewed. Log transformation suggested.")

    result = {
        "success": True,
        "insights": insights,
        "has_issues": len(insights) > 0
    }
    
    print("\n--- 9. AUTOMATED AI INSIGHTS ---")
    if insights:
        for insight in insights:
            print(f"- {insight}")
    else:
        print("No major issues detected. Your data looks good for modeling!")
    
    print("-" * 30)
    return result


###########################run_eda############################
def run_full_eda(df, target_col=None):
    """
    Executes the complete 9-task EDA pipeline.
    Returns a dictionary with all results.
    """
    print("\n" + "="*50)
    print("      STARTING AUTOMATED EDA PIPELINE      ")
    print("="*50)
    
    # Run all tasks and collect results
    results = {}
    
    results['basic_info'] = get_basic_info(df)
    results['statistical_summary'] = get_statistical_summary(df)
    results['missing_values'] = get_missing_values_analysis(df)
    results['unique_values'] = get_unique_values_analysis(df)
    results['correlation'] = get_correlation_analysis(df)
    results['distribution'] = get_distribution_analysis(df)
    results['outliers'] = get_outlier_analysis(df)
    
    # Task 8: Target Analysis
    if target_col:
        results['target_analysis'] = get_target_analysis(df, target_col)
    else:
        results['target_analysis'] = get_target_analysis(df, df.columns[-1])
    
    # Task 9: Insights
    results['insights'] = get_automated_insights(df)
    
    print("\n" + "="*50)
    print("          EDA PIPELINE COMPLETE           ")
    print("="*50)
    
    return results