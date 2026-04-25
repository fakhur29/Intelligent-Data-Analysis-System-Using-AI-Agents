from datetime import datetime

def generate_report(problem_type, evaluation, best_model, best_score, best_params, 
                    selected_features, removed_features, dataset_shape, target_col):
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print("\n" + "="*60)
    print("         AI DATA ANALYSIS REPORT")
    print("="*60)
    
    print(f"\nGenerated: {timestamp}")
    print(f"Problem Type: {problem_type.upper()}")
    print(f"Target Column: {target_col}")
    
    print("\n--- DATASET OVERVIEW ---")
    print(f"Total Rows: {dataset_shape[0]}")
    print(f"Total Columns: {dataset_shape[1]}")
    print(f"Selected Features: {len(selected_features)}")
    print(f"Removed Features: {len(removed_features)}")
    if removed_features:
        print(f"  Removed: {removed_features}")
    
    print("\n--- BEST MODEL ---")
    print(f"Model: {best_model}")
    if problem_type == "classification":
        print(f"Accuracy: {best_score:.4f}")
    else:
        print(f"R2-Score: {best_score:.4f}")
    
    print("\nBest Parameters:")
    for k, v in best_params.items():
        param_name = k.replace('model__', '')
        print(f"  {param_name}: {v}")
    
    print("\n--- ALL MODELS PERFORMANCE ---")
    for model_name, metrics in evaluation.items():
        print(f"\n{model_name}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
    
    print("\n--- RECOMMENDATIONS ---")
    if best_score >= 0.9:
        print("✅ Excellent performance. Ready for production.")
    elif best_score >= 0.8:
        print("✅ Good performance. Can be used with monitoring.")
    elif best_score >= 0.7:
        print("⚠️  Acceptable performance. Consider improvements.")
    else:
        print("❌ Poor performance. Needs more data or features.")
    
    print("\n" + "="*60)
    
    return {
        "success": True,
        "timestamp": timestamp,
        "problem_type": problem_type,
        "target_column": target_col,
        "best_model": best_model,
        "best_score": best_score,
        "best_parameters": best_params,
        "selected_features": selected_features,
        "removed_features": removed_features,
        "dataset_shape": dataset_shape,
        "all_models": evaluation
    }