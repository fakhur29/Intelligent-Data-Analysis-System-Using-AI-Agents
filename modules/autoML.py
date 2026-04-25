import pandas as pd
import numpy as np
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
################################select_best_model#######################

def select_best_model(evaluation, problem_type):
    
    ranked_models = []
    
    if problem_type == "classification":
        metric_key = "Accuracy"
    else:
        metric_key = "R2-Score"
    
    for model_name, metrics in evaluation.items():
        score = metrics[metric_key]
        ranked_models.append({
            "model": model_name,
            "score": score
        })
    
    ranked_models.sort(key=lambda x: x['score'], reverse=True)
    
    best_model = ranked_models[0]['model']
    best_score = ranked_models[0]['score']
    
    print(f"✅ Best Model Selected")
    print(f"   Model: {best_model}")
    print(f"   {metric_key}: {best_score}")
    print(f"\n📊 All Models Ranked:")
    for i, item in enumerate(ranked_models, 1):
        print(f"   {i}. {item['model']}: {item['score']}")
    
    return {
        "success": True,
        "best_model": best_model,
        "best_score": best_score,
        "all_models_ranked": ranked_models
    }

################################auto_feature_selection##########################
# from sklearn.feature_selection import SelectKBest, f_regression, f_classif # type: ignore


# def auto_feature_selection(X, y, problem_type="regression"):
    
#     if problem_type == "classification":
#         selector = SelectKBest(score_func=f_classif, k='all')
#     else:
#         selector = SelectKBest(score_func=f_regression, k='all')
#     # X = pd.get_dummies(X) #this line added for gpt
#     selector.fit(X, y)
    
#     scores = selector.scores_
#     feature_scores = list(zip(X.columns, scores))
#     feature_scores.sort(key=lambda x: x[1], reverse=True)
    
#     threshold = np.mean(scores)
#     selected_features = [col for col, score in feature_scores if score >= threshold]
#     removed_features = [col for col, score in feature_scores if score < threshold]
    
#     X_selected = X[selected_features]
    
#     print(f"✅ Feature Selection Complete")
#     print(f"   Original Features: {len(X.columns)}")
#     print(f"   Selected Features: {len(selected_features)}")
#     print(f"   Removed Features: {len(removed_features)}")
#     if removed_features:
#         print(f"   Removed: {removed_features}")
    
#     return {
#         "success": True,
#         "original_features": len(X.columns),
#         "selected_features": len(selected_features),
#         "removed_features": removed_features,
#         "X_selected": X_selected,
#         "feature_names": selected_features
#     }

################################auto_feature_selection##########################
from sklearn.feature_selection import SelectKBest, f_regression, f_classif

# def auto_feature_selection(X, y, problem_type="regression"):
#     # --- ADD THIS LINE TO FIX THE ERROR ---
#     # This converts 'male'/'female' into 1s and 0s just for the selection math
#     X_encoded = pd.get_dummies(X, drop_first=True) 
    
#     if problem_type == "classification":
#         selector = SelectKBest(score_func=f_classif, k='all')
#     else:
#         selector = SelectKBest(score_func=f_regression, k='all')
    
#     # Fit on the ENCODED data, not the raw text data
#     selector.fit(X_encoded, y)
    
#     scores = selector.scores_
#     # Note: We use X_encoded.columns because get_dummies might have created new column names
#     feature_scores = list(zip(X_encoded.columns, scores))
#     feature_scores.sort(key=lambda x: x[1], reverse=True)
    
#     threshold = np.percentile(scores, 60)  # keep top 50%
#     # We find which of the ORIGINAL columns are important
#     # selected_features = [col for col in X.columns if any(col in enc_f for enc_f, s in feature_scores if s >= threshold)]
#     selected_encoded = [f for f, s in feature_scores if s >= threshold]
#     selected_features = set()
#     for enc_col in selected_encoded:
#         base_col = enc_col.split('_')[0]
#         if base_col in X.columns:
#             selected_features.add(base_col)

#     selected_features = list(selected_features)
#     # Remove known problematic columns
#     bad_features = []

#     for col in selected_features:
#         # Remove high-missing columns (only if NOT important)
#         if X[col].isnull().mean() > 0.6:
#             bad_features.append(col)
        
#         # Remove high-cardinality categorical
#         elif X[col].dtype == 'object':
#             unique_ratio = X[col].nunique() / len(X)
#             if unique_ratio > 0.85:
#                 bad_features.append(col)

#     # Apply removal
#     selected_features = [col for col in selected_features if col not in bad_features]
#     removed_features = [col for col in X.columns if col not in selected_features]
    
#     X_selected = X[selected_features]
    
    
#     print(f"✅ Feature Selection Complete")
#     print(f"   Original Features: {len(X.columns)}")
#     print(f"   Selected Features: {len(selected_features)}")
#     print(f"   Removed Features: {len(removed_features)}")
#     if removed_features:
        
#         print(f"   Removed: {removed_features}")
#     return {
#         "success": True,
#         "X_selected": X_selected,
#         "feature_names": selected_features,
#         "removed_features": removed_features
#     }


from sklearn.feature_selection import SelectKBest, f_classif, f_regression, mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# def auto_feature_selection(X, y, problem_type="regression"):
    
#     # 1. NEW BLOCK: Prepare y (Target) 
#     # This prevents the "could not convert string to float" error
#     y_encoded = y.copy()
#     if problem_type == "classification" and not pd.api.types.is_numeric_dtype(y):
#         le_y = LabelEncoder()
#         y_encoded = le_y.fit_transform(y.astype(str))
    
#     # 2. Prepare X (Features)
#     X_encoded = X.copy()
#     numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
#     categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
#     for col in categorical_cols:
#         le = LabelEncoder()
#         X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
        
#     # 3. Setup Selectors
#     all_scores = pd.DataFrame(index=X.columns)
    
#     if problem_type == "classification":
#         mi_selector = SelectKBest(score_func=mutual_info_classif, k='all')
#         f_selector = SelectKBest(score_func=f_classif, k='all')
#     else:
#         mi_selector = SelectKBest(score_func=mutual_info_regression, k='all')
#         f_selector = SelectKBest(score_func=f_regression, k='all')
    
#     # 4. Fit using the newly encoded y_encoded
#     mi_selector.fit(X_encoded, y_encoded)
#     f_selector.fit(X_encoded, y_encoded)
    
#     all_scores['mutual_info'] = mi_selector.scores_
#     all_scores['f_score'] = f_selector.scores_
    
#     # 5. Correlation calculation using y_encoded
#     for col in numeric_cols:
#         # We ensure y_encoded is numeric before correlation
#         corr = abs(X_encoded[col].corr(pd.Series(y_encoded)))
#         all_scores.loc[col, 'correlation'] = corr
    
#     all_scores['correlation'] = all_scores['correlation'].fillna(0)
    
#     # 6. Random Forest (also uses y_encoded)
#     rf_model = RandomForestClassifier(n_estimators=50, random_state=42) if problem_type == "classification" else RandomForestRegressor(n_estimators=50, random_state=42)
#     rf_model.fit(X_encoded, y_encoded)
#     all_scores['rf_importance'] = rf_model.feature_importances_
    
#     # ... (Rest of your code remains exactly the same from here down)
    
#     normalized_scores = all_scores.copy()
#     for col in normalized_scores.columns:
#         min_val = normalized_scores[col].min()
#         max_val = normalized_scores[col].max()
#         if max_val - min_val > 0:
#             normalized_scores[col] = (normalized_scores[col] - min_val) / (max_val - min_val)
#         else:
#             normalized_scores[col] = 0
    
#     final_score = (
#         normalized_scores['mutual_info'] * 0.3 +
#         normalized_scores['f_score'] * 0.2 +
#         normalized_scores['correlation'] * 0.2 +
#         normalized_scores['rf_importance'] * 0.3
#     )
    
#     all_scores['final_score'] = final_score
#     all_scores_sorted = all_scores.sort_values('final_score', ascending=False)
    
#     min_features = max(3, len(X.columns) // 2)
#     score_threshold = np.percentile(final_score, 40)
    
#     selected_features = all_scores_sorted[all_scores_sorted['final_score'] >= score_threshold].index.tolist()
    
#     if len(selected_features) < min_features:
#         selected_features = all_scores_sorted.head(min_features).index.tolist()
    
#     removed_features = [col for col in X.columns if col not in selected_features]
    
#     X_selected = X[selected_features]
    
#     return {
#         "success": True,
#         "X_selected": X_selected,
#         "feature_names": selected_features,
#         "removed_features": removed_features,
#         "feature_scores": all_scores_sorted.to_dict()
#     }

def auto_feature_selection(X, y, problem_type="regression"):
    
    # 1. Prepare y (Target)
    y_encoded = y.copy()
    if problem_type == "classification" and not pd.api.types.is_numeric_dtype(y):
        le_y = LabelEncoder()
        y_encoded = le_y.fit_transform(y.astype(str))
    
    # 2. NEW: Remove high-missing columns (important fix)
    high_missing_cols = [col for col in X.columns if X[col].isnull().mean() > 0.6]
    X = X.drop(columns=high_missing_cols)

    # 3. Prepare X (Features)
    numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # 🔥 FIX: Use One-Hot Encoding instead of LabelEncoder
    X_encoded = pd.get_dummies(X, drop_first=True)
    
    # 4. Setup Selectors
    all_scores = pd.DataFrame(index=X_encoded.columns)
    
    if problem_type == "classification":
        mi_selector = SelectKBest(score_func=mutual_info_classif, k='all')
        f_selector = SelectKBest(score_func=f_classif, k='all')
    else:
        mi_selector = SelectKBest(score_func=mutual_info_regression, k='all')
        f_selector = SelectKBest(score_func=f_regression, k='all')
    
    # 5. Fit
    mi_selector.fit(X_encoded, y_encoded)
    f_selector.fit(X_encoded, y_encoded)
    
    all_scores['mutual_info'] = mi_selector.scores_
    all_scores['f_score'] = f_selector.scores_
    
    # 6. Correlation (only for numeric original columns)
    for col in numeric_cols:
        if col in X_encoded.columns:
            corr = abs(X_encoded[col].corr(pd.Series(y_encoded)))
            all_scores.loc[col, 'correlation'] = corr
    
    all_scores['correlation'] = all_scores['correlation'].fillna(0)
    
    # 7. Random Forest
    rf_model = RandomForestClassifier(n_estimators=50, random_state=42) if problem_type == "classification" else RandomForestRegressor(n_estimators=50, random_state=42)
    rf_model.fit(X_encoded, y_encoded)
    all_scores['rf_importance'] = rf_model.feature_importances_
    
    # 8. Normalize Scores
    normalized_scores = all_scores.copy()
    for col in normalized_scores.columns:
        min_val = normalized_scores[col].min()
        max_val = normalized_scores[col].max()
        if max_val - min_val > 0:
            normalized_scores[col] = (normalized_scores[col] - min_val) / (max_val - min_val)
        else:
            normalized_scores[col] = 0
    
    final_score = (
        normalized_scores['mutual_info'] * 0.3 +
        normalized_scores['f_score'] * 0.2 +
        normalized_scores['correlation'] * 0.2 +
        normalized_scores['rf_importance'] * 0.3
    )
    
    # 🔥 FIX: Aggregate encoded features back to original features
    feature_scores = {}
    for col in X.columns:
        related_cols = [c for c in X_encoded.columns if c.startswith(col + "_") or c == col]
        if related_cols:
            feature_scores[col] = final_score[related_cols].max()
    
    feature_scores = pd.Series(feature_scores).sort_values(ascending=False)
    
    # 9. Selection Logic
    min_features = max(4, len(X.columns) // 2)
    score_threshold = np.percentile(feature_scores.values, 40)
    
    selected_features = feature_scores[feature_scores >= score_threshold].index.tolist()
    
    if len(selected_features) < min_features:
        selected_features = feature_scores.head(min_features).index.tolist()
    
    removed_features = [col for col in X.columns if col not in selected_features]
    
    X_selected = X[selected_features]
    
    return {
        "success": True,
        "X_selected": X_selected,
        "feature_names": selected_features,
        "removed_features": removed_features,
        "feature_scores": feature_scores.to_dict()
    }
#############################auto_hyperparameter_tuning##########################\

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

def auto_hyperparameter_tuning(best_model_name, X_train, y_train, problem_type):
    
    # Parameter grids (UPDATED with model__)
    param_grids = {
        "Logistic Regression": {
            'model__C': [0.1, 1, 10],
            'model__penalty': ['l2']
        },
        "Decision Tree": {
            'model__max_depth': [5, 10, 15, 20],
            'model__min_samples_split': [2, 5, 10]
        },
        "Decision Tree Regressor": {
            'model__max_depth': [5, 10, 15, 20],
            'model__min_samples_split': [2, 5, 10]
        },
        "Random Forest": {
            'model__n_estimators': [50, 100, 150],
            'model__max_depth': [5, 10, 15, None]
        },
        "Random Forest Regressor": {
            'model__n_estimators': [50, 100, 150],
            'model__max_depth': [5, 10, 15, None]
        },
        "Linear Regression": {}
    }
    
    # Model mapping
    model_map = {
        "Logistic Regression": LogisticRegression(max_iter=1000, solver='lbfgs', random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
        "Random Forest Regressor": RandomForestRegressor(random_state=42),
        "Linear Regression": LinearRegression()
    }
    
    if best_model_name not in model_map:
        return {
            "success": False,
            "error": f"Model '{best_model_name}' not supported"
        }
    
    base_model = model_map[best_model_name]
    param_grid = param_grids.get(best_model_name, {})
    
    # ✅ Build Pipeline here
    pipeline = build_pipeline(base_model, X_train)
    
    # If no tuning required
    if not param_grid:
        pipeline.fit(X_train, y_train)
        return {
            "success": True,
            "tuned_model": pipeline,
            "best_parameters": {},
            "message": "No tuning needed for this model"
        }
    
    # Scoring
    scoring = 'accuracy' if problem_type == 'classification' else 'r2'
    
    # ✅ Use pipeline in GridSearch
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=3,
        scoring=scoring
    )
    
    grid_search.fit(X_train, y_train)
    
    tuned_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    print(f"✅ Hyperparameter Tuning Complete")
    print(f"   Model: {best_model_name}")
    print(f"   Best Parameters: {best_params}")
    print(f"   Best CV Score: {best_score:.4f}")
    
    return {
        "success": True,
        "tuned_model": tuned_model,
        "best_parameters": best_params,
        "best_score": best_score
    }

#################################generate_model_summary###########################

def generate_model_summary(best_model_name, best_score, all_models_ranked, problem_type):
    
    if problem_type == "classification":
        metric_name = "Accuracy"
    else:
        metric_name = "R2-Score"
    
    
    if len(all_models_ranked) > 1:
        second_best_model = all_models_ranked[1]['model']
        second_best_score = all_models_ranked[1]['score']
    else:
        second_best_model = "N/A"
        second_best_score = 0
    
    difference = best_score - second_best_score
    
    summary_text = f"{best_model_name} achieved {metric_name} of {best_score:.4f}, outperforming {second_best_model} ({second_best_score:.4f}) by {difference:.4f}."
    
    if best_score >= 0.9:
        recommendation = "This model is excellent and recommended for production."
    elif best_score >= 0.8:
        recommendation = "This model is good and can be used with monitoring."
    elif best_score >= 0.7:
        recommendation = "This model is acceptable but could be improved."
    else:
        recommendation = "This model needs improvement. Consider more data or feature engineering."
    
    print(f"✅ Model Summary Generated")
    print(f"   Best Model: {best_model_name}")
    print(f"   {metric_name}: {best_score:.4f}")
    print(f"   Summary: {summary_text}")
    print(f"   Recommendation: {recommendation}")
    
    return {
        "success": True,
        "model_name": best_model_name,
        "performance_metric": metric_name,
        "score": best_score,
        "summary": summary_text,
        "recommendation": recommendation
    }

#################################run_automl_pipeline##########################
def run_automl_pipeline(models, evaluation, X_train, y_train, problem_type):
    
    # Step 1: Select Best Model
    best_model_result = select_best_model(evaluation, problem_type)
    best_model_name = best_model_result['best_model']
    best_score = best_model_result['best_score']
    all_models_ranked = best_model_result['all_models_ranked']
    
    # Step 2: Feature Selection
    feature_result = auto_feature_selection(X_train, y_train, problem_type)
    X_train_selected = feature_result['X_selected']
    
    # Step 3: Hyperparameter Tuning
    tuning_result = auto_hyperparameter_tuning(
        best_model_name, 
        X_train_selected, 
        y_train, 
        problem_type
    )
    
    tuned_model = tuning_result['tuned_model']
    best_parameters = tuning_result['best_parameters']
    
    # 🔴 Critical Fix: Ensure model is trained on selected features
    if tuned_model is not None:
        tuned_model.fit(X_train_selected, y_train)
    
    # Step 4: Generate Summary
    summary_result = generate_model_summary(
        best_model_name, 
        best_score, 
        all_models_ranked, 
        problem_type
    )
    
    print(f"\n✅ AutoML Pipeline Complete")
    
    return {
        "success": True,
        "best_model": tuned_model,
        "best_model_name": best_model_name,
        "best_score": best_score,
        "best_parameters": best_parameters,
        "selected_features": feature_result['feature_names'],
        "removed_features": feature_result['removed_features'],
        "all_models_ranked": all_models_ranked,
        "summary": summary_result['summary'],
        "recommendation": summary_result['recommendation']
    }