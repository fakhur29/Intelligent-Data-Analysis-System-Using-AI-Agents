import pandas as pd
from sklearn import pipeline
import numpy as np
from sklearn.model_selection import train_test_split


from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def build_pipeline(model, X):

    numeric_cols = X.select_dtypes(include=['number']).columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ]
    )

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    return pipeline

 ########################### detect_problem_type ##########################
def detect_problem_type(df, target_col):
      
    # Check if target column exists
    if target_col not in df.columns:
        return {
            "success": False,
            "error": f"Target column '{target_col}' not found in dataset"
        }
    
    target = df[target_col]
    
    # Get info about target
    dtype = target.dtype
    unique_count = target.nunique()
    
    # Decision Logic
    if dtype == 'object':
        # Text/String type → Always Classification
        problem_type = "classification"
        explanation = f"Target is text type (object). This is a Classification problem."
    
    elif pd.api.types.is_numeric_dtype(target):
        # Numeric type → Check unique values
        if unique_count < 20:
            problem_type = "classification"
            explanation = f"Target has only {unique_count} unique values. Treating as Classification."
        else:
            problem_type = "regression"
            explanation = f"Target has {unique_count} unique values. Treating as Regression."
    
    else:
        # Unknown type → Default to Regression (safest choice)
        problem_type = "regression"
        explanation = f"Unknown data type. Defaulting to Regression."
    
    result = {
        "success": True,
        "problem_type": problem_type,
        "target_column": target_col,
        "unique_values": unique_count,
        "data_type": str(dtype),
        "explanation": explanation
    }
    
    print(f"✅ Problem Type Detected: {problem_type.upper()}")
    print(f"   Target: {target_col}")
    print(f"   Unique Values: {unique_count}")
    print(f"   Data Type: {dtype}")
    print(f"   Reason: {explanation}")
    
    return result
##########################split_train_test ##########################



def split_data(df, target_col,problem_type):
    
    
    if target_col not in df.columns:
        return {
            "success": False,
            "error": f"Target column '{target_col}' not found"
        }
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    if problem_type == "classification":

        X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
    )
    
    result = {
        "success": True,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "feature_count": X.shape[1],
        "total_samples": len(df)
    }
    
    print(f"✅ Data Split Complete")
    print(f"   Total Samples: {len(df)}")
    print(f"   Train Set: {len(X_train)} (80%)")
    print(f"   Test Set: {len(X_test)} (20%)")
    print(f"   Features: {X.shape[1]}")
    
    return result

###############################train_classification_models##########################

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

def train_classification_models(X_train, y_train):
    
    models = {}
    
    lr = LogisticRegression(max_iter=1000, random_state=42)
    # lr.fit(X_train, y_train)
    # lr_pipeline = build_pipeline(lr)
    lr_pipeline = build_pipeline(lr, X_train)
    lr_pipeline.fit(X_train, y_train)
    models["Logistic Regression"] = lr_pipeline
    # models["Logistic Regression"] = lr
    
    dt = DecisionTreeClassifier(random_state=42)
    # dt.fit(X_train, y_train)
    dt_pipeline = build_pipeline(dt, X_train)
    dt_pipeline.fit(X_train, y_train)
    models["Decision Tree"] = dt_pipeline

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    # rf.fit(X_train, y_train)
    rf_pipeline = build_pipeline(rf, X_train)
    rf_pipeline.fit(X_train, y_train)
    models["Random Forest"] = rf_pipeline
    
    print(f"✅ Classification Models Trained")
    print(f"   Logistic Regression ✓")
    print(f"   Decision Tree ✓")
    print(f"   Random Forest ✓")
    
    return {
        "success": True,
        "models": models,
        "model_count": len(models)
    }

#################################train_regression_models##########################

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

def train_regression_models(X_train, y_train):
    
    models = {}
    
    lr = LinearRegression()
    lr_pipeline = build_pipeline(lr,X_train)
    lr_pipeline.fit(X_train, y_train)
    models["Linear Regression"] = lr_pipeline
    
    dt = DecisionTreeRegressor(random_state=42)
    dt_pipeline = build_pipeline(dt, X_train)
    dt_pipeline.fit(X_train, y_train)
    models["Decision Tree Regressor"] = dt_pipeline

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_pipeline = build_pipeline(rf,X_train)
    rf_pipeline.fit(X_train, y_train)
    models["Random Forest Regressor"] = rf_pipeline
    
    print(f"✅ Regression Models Trained")
    print(f"   Linear Regression ✓")
    print(f"   Decision Tree Regressor ✓")
    print(f"   Random Forest Regressor ✓")
    
    return {
        "success": True,
        "models": models,
        "model_count": len(models)
    }



#######################################evaluate_classification_models##########################

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_classification_models(models, X_test, y_test):
    
    evaluation = {}
    
    for model_name, model in models.items():
        y_pred = model.predict(X_test)
        
        print(f"\n{model_name}:")
        print(f"  Predictions distribution: {np.unique(y_pred, return_counts=True)}")
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred,average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred,average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted',zero_division=0)
        if accuracy == precision == recall == f1:
            print(f"  ⚠️ WARNING: All metrics identical!")
        
        evaluation[model_name] = {
            'Accuracy': round(accuracy, 4),
            'Precision': round(precision, 4),
            'Recall': round(recall, 4),
            'F1-Score': round(f1, 4)
        }
    
    print(f"✅ Classification Models Evaluated")
    for model_name, metrics in evaluation.items():
        print(f"\n{model_name}:")
        for metric, value in metrics.items():
            print(f"   {metric}: {value}")
    
    return {
        "success": True,
        "evaluation": evaluation
    }


#######################################evaluate_regression_models##########################

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def evaluate_regression_models(models, X_test, y_test):
    
    evaluation = {}
    
    for model_name, model in models.items():
        y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        evaluation[model_name] = {
            'RMSE': round(rmse, 4),
            'MAE': round(mae, 4),
            'R2-Score': round(r2, 4)
        }
    
    print(f"✅ Regression Models Evaluated")
    for model_name, metrics in evaluation.items():
        print(f"\n{model_name}:")
        for metric, value in metrics.items():
            print(f"   {metric}: {value}")
    
    return {
        "success": True,
        "evaluation": evaluation
    }