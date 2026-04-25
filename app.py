import streamlit as st
import pandas as pd
import pickle
import io
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / "backend"))

from modules.loader import load_file
from modules.data_cleaner import (handle_missing_values, remove_duplicates, 
                                   format_correction, fix_data_types, 
                                   handle_outliers, drop_irrelevant_columns)
from modules.eda import run_full_eda
from modules.train_modules import (detect_problem_type, split_data, 
                                    train_classification_models, train_regression_models,
                                    evaluate_classification_models, evaluate_regression_models)
from modules.autoML import (select_best_model, auto_feature_selection, 
                            auto_hyperparameter_tuning, generate_model_summary, 
                            run_automl_pipeline)
from modules.prediction import predict_new_data
from modules.generate_report import generate_report

st.set_page_config(page_title="AutoML Data Analysis Tool", layout="wide", page_icon="🤖")

st.markdown("""
    <style>
    .main-title {
        text-align: center;
        color: #0ba803;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🤖 AutoML Data Analysis Tool</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload your dataset and let AI do the rest</div>', unsafe_allow_html=True)

if "analysis_complete" not in st.session_state:
    st.session_state.analysis_complete = False
if "uploaded_df" not in st.session_state:
    st.session_state.uploaded_df = None
if "target_selected" not in st.session_state:
    st.session_state.target_selected = None
if "pipeline_results" not in st.session_state:
    st.session_state.pipeline_results = None

st.markdown("---")

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    uploaded_file = st.file_uploader("📂 Upload your dataset", type=['csv', 'xlsx', 'json'])

if uploaded_file is not None:

    try:
        
        if uploaded_file.name.endswith('.csv'):
            try:
                st.session_state.uploaded_df = pd.read_csv(uploaded_file)
            except UnicodeDecodeError:
                st.session_state.uploaded_df = pd.read_csv(uploaded_file, encoding='latin-1')
            
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            st.session_state.uploaded_df = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith('.json'):
            st.session_state.uploaded_df = pd.read_json(uploaded_file)
        st.success("✅ File uploaded successfully!")
        st.session_state.dataset_name = uploaded_file.name.split('.')[0]
        
        
        st.markdown("---")
        
        columns = st.session_state.uploaded_df.columns.tolist()
        target_col = st.selectbox("🎯 Select Target Column (what to predict)", columns)
        st.session_state.target_selected = target_col
        
        st.markdown("---")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            start_button = st.button("🚀 Start Analyzing",)
        
        if start_button:
            st.session_state.analysis_complete = False
            
            with st.spinner("Processing your data..."):
                    
                    status_text = st.empty()
                    
                     
                    df = st.session_state.uploaded_df.copy()

                    status_text.text("🧹 Cleaning data...")                    
                    progress_bar = st.progress(10)                    
                    df = handle_missing_values(df)
                    df = remove_duplicates(df)
                    df = format_correction(df)
                    df = fix_data_types(df)
                    df = handle_outliers(df)
                    df = drop_irrelevant_columns(df)

                    status_text.text("📊 Running EDA...")
                    progress_bar.progress(25)
                    eda_results = run_full_eda(df, target_col=target_col)
                    
                    status_text.text("🎯 Detecting problem type...")
                    progress_bar.progress(40)
                    problem_detection = detect_problem_type(df, target_col)
                    problem_type = problem_detection['problem_type']
                    
                    status_text.text("✂️ Splitting data...")
                    progress_bar.progress(50)                    
                    split_result = split_data(df, target_col, problem_type)
                    X_train = split_result['X_train']
                    X_test = split_result['X_test']
                    y_train = split_result['y_train']
                    y_test = split_result['y_test']

                    status_text.text("🤖 Training models...")
                    progress_bar.progress(60)
                    if problem_type == 'classification':
                        models_result = train_classification_models(X_train, y_train)
                        evaluation_result = evaluate_classification_models(models_result['models'], X_test, y_test)
                    else:
                        models_result = train_regression_models(X_train, y_train)
                        evaluation_result = evaluate_regression_models(models_result['models'], X_test, y_test)
                    
                    models = models_result['models']
                    evaluation = evaluation_result['evaluation']

                    status_text.text("🧠 Running AutoML...")
                    progress_bar.progress(75)
                    pipeline_result = run_automl_pipeline(models, evaluation, X_train, y_train, problem_type)
                    
                    status_text.text("📝 Generating report...")
                    progress_bar.progress(90)
                    best_model = pipeline_result['best_model']
                    best_model_name = pipeline_result['best_model_name']
                    best_score = pipeline_result['best_score']
                    best_parameters = pipeline_result['best_parameters']
                    selected_features = pipeline_result['selected_features']
                    removed_features = pipeline_result['removed_features']
                    
                    report_result = generate_report(
                        problem_type=problem_type,
                        evaluation=evaluation,
                        best_model=best_model_name,
                        best_score=best_score,
                        best_params=best_parameters,
                        selected_features=selected_features,
                        removed_features=removed_features,
                        dataset_shape=df.shape,
                        target_col=st.session_state.target_selected
                    )
                    status_text.text("💾 Saving results...")
                    progress_bar.progress(100)
                    
                    st.session_state.pipeline_results = {
                        'uploaded_df': st.session_state.uploaded_df,                   
                        'df': df,
                        'eda_results': eda_results,
                        'best_model': best_model,
                        'best_model_name': best_model_name,
                        'best_score': best_score,
                        'best_parameters': best_parameters,
                        'selected_features': selected_features,
                        'removed_features': removed_features,
                        'problem_type': problem_type,
                        'evaluation': evaluation,
                        'report': report_result,
                        'X_test': X_test,
                        'y_test': y_test,
                        'X_train': X_train,
                        'target_col': target_col
                    }
                    
                    st.session_state.analysis_complete = True
                
            st.success("✅ Analysis Complete!")
        
        if st.session_state.analysis_complete and st.session_state.pipeline_results:
            results = st.session_state.pipeline_results
            
            st.markdown("---")
            st.markdown("## 📊 Analysis Results")
            
            with st.expander("📋 View Data", expanded=False):
                st.subheader("Dataset Preview (First 10 rows)")
                st.dataframe(results['uploaded_df'].head(10), width='stretch')
                st.write(f"**Total Rows:** {results['uploaded_df'].shape[0]}")
                st.write(f"**Total Columns:** {results['uploaded_df'].shape[1]}")
            
            with st.expander("🤖 Model Report", expanded=False):
                st.write(results['report'])
            
            with st.expander("📈 EDA Report", expanded=False):
                st.write("Exploratory Data Analysis completed. Key findings:")
                st.write(results['eda_results'])
            
            with st.expander("🔮 Make Predictions", expanded=False):
                prediction_type = st.radio("Choose prediction method:", 
                                          ["Upload New Dataset", "Single Prediction"],
                                          horizontal=True)
                
                if prediction_type == "Upload New Dataset":
                    new_file = st.file_uploader("📂 Upload new dataset for predictions", type=['csv', 'xlsx', 'json'])
                    if new_file is not None:
                        try:
                            if new_file.name.endswith('.csv'):
                                X_new = pd.read_csv(new_file)
                            elif new_file.name.endswith(('.xlsx', '.xls')):
                                X_new = pd.read_excel(new_file)
                            elif new_file.name.endswith('.json'):
                                X_new = pd.read_json(new_file)
                            X_new_selected = X_new[results['selected_features']]
                            
                            if st.button("🔮 Predict Now", key="predict_new"):
                                prediction_result = predict_new_data(results['best_model'], X_new_selected)
                                
                                if prediction_result['success']:
                                    st.success(f"✅ Predictions made for {len(X_new_selected)} rows")
                                    
                                    predictions_df = X_new_selected.copy()
                                    predictions_df['Prediction'] = prediction_result['predictions']
                                    
                                    st.dataframe(predictions_df, width='stretch')
                                    
                                    csv = predictions_df.to_csv(index=False)
                                    st.download_button(
                                        label="⬇️ Download Predictions (CSV)",
                                        data=csv,
                                        file_name="predictions.csv",
                                        mime="text/csv"
                                    )
                                else:
                                    st.error(f"❌ Prediction failed: {prediction_result['error']}")
                        except Exception as e:
                            st.error(f"❌ Error loading file: {str(e)}")
                
                else:
                    st.write("**Enter feature values for single prediction:**")
                    
                    input_data = {}
                    feature_cols = st.columns(2)
                    
                    for idx, feature in enumerate(results['selected_features']):
                        col_idx = idx % 2
                        
                        feature_dtype = results['X_train'][feature].dtype
                        
                        if feature_dtype in ['int64', 'int32']:
                            input_data[feature] = feature_cols[col_idx].number_input(
                                f"{feature} (Integer)",
                                value=int(results['X_train'][feature].mean()),
                                step=1
                            )
                        elif feature_dtype in ['float64', 'float32']:
                            input_data[feature] = feature_cols[col_idx].number_input(
                                f"{feature} (Float)",
                                value=float(results['X_train'][feature].mean()),
                                step=0.01
                            )
                        else:
                            unique_vals = results['X_train'][feature].unique().tolist()
                            input_data[feature] = feature_cols[col_idx].selectbox(
                                f"{feature} (Category)",
                                unique_vals
                            )
                    
                    if st.button("🔮 Predict", key="predict_single"):
                        try:
                            X_single = pd.DataFrame([input_data])
                            prediction = results['best_model'].predict(X_single)[0]
                            
                            st.success(f"✅ Prediction: **{prediction}**")
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
            
            st.markdown("---")
            st.markdown("## 📥 Download Options")
            
            col1, col2, col3 = st.columns(3)
            
            from datetime import datetime

            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            model_name_safe = results['best_model_name'].lower().replace(" ", "_")
            dataset_name = st.session_state.dataset_name

            with col1:
                report_text = str(results['report'])
                st.download_button(
                    label="📊 Download Model Report",
                    data=report_text,
                    file_name=f"{dataset_name}_{model_name_safe}_model_report_{timestamp}.txt",
                    mime="text/plain"
                )

            with col2:
                st.download_button(
                    label="📈 Download EDA Report",
                    data=str(results['eda_results']),
                    file_name=f"{dataset_name}_{model_name_safe}_eda_report_{timestamp}.txt",
                    mime="text/plain"
                )

            with col3:
                model_pkl = pickle.dumps(results['best_model'])
                st.download_button(
                    label="💾 Save Model (PKL)",
                    data=model_pkl,
                    file_name=f"{dataset_name}_{model_name_safe}_best_model_{timestamp}.pkl",
                    mime="application/octet-stream"
                )
            
            st.markdown("---")
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("🔄 Start Over (New Analysis)", width='stretch'):
                    st.session_state.clear()
                    st.rerun()

    except Exception as e:
     st.error(f"❌ Error: {str(e)}")