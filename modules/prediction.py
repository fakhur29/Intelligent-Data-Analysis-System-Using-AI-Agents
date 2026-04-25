def predict_new_data(pipeline, new_data):
    
    try:
        if new_data.empty:
            return {"success": False, "error": "new_data is empty"}
        
        predictions = pipeline.predict(new_data)
        
        print("✅ Prediction completed successfully")
        
        return {
            "success": True,
            "predictions": predictions,
            "rows_predicted": len(predictions)
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }