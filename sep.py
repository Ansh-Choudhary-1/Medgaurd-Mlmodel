import os
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import io

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the trained model
MODEL_PATH = "lstm_model.keras"  # Update with your model path
model = tf.keras.models.load_model(MODEL_PATH)

def preprocess_psv_file(file_content):
    """Preprocess the PSV file for prediction."""
    # Read the PSV file from the uploaded content
    df = pd.read_csv(io.StringIO(file_content.decode('utf-8')), delimiter='|')
    
    # Fill missing values
    df_filled = df.copy()
    df_filled = df_filled.fillna(method='ffill').fillna(method='bfill')
    
    # Drop columns with all NaN values
    useless_cols = df_filled.columns[df_filled.isnull().sum() == len(df_filled)]
    df_filled.drop(columns=useless_cols, inplace=True)
    
    # Impute remaining missing values with median
    df_clean = df_filled.fillna(df_filled.median())
    
    # Normalize numeric features (excluding SepsisLabel if present)
    feature_cols = df_clean.columns.tolist()
    if "SepsisLabel" in feature_cols:
        feature_cols.remove("SepsisLabel")
    
    scaler = StandardScaler()
    df_clean[feature_cols] = scaler.fit_transform(df_clean[feature_cols])
    
    return df_clean

def create_sequences(data, window_size=1):
    """Create time series sequences from data."""
    X_seq = []
    for i in range(len(data) - window_size + 1):
        X_seq.append(data[i:i+window_size])
    return np.array(X_seq)

@app.route('/api/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not file.filename.endswith('.psv'):
        return jsonify({'error': 'File must be a PSV file'}), 400
    
    try:
        # Read and preprocess the file
        file_content = file.read()
        df_clean = preprocess_psv_file(file_content)
        
        # Convert to feature array
        features = df_clean.values
        
        # Create sequences for LSTM input (window_size=1 as per your model)
        X_seq = create_sequences(features, window_size=1)
        
        # Make predictions
        predictions = model.predict(X_seq)
        
        # Process predictions
        risk_scores = predictions.flatten().tolist()
        
        # Map predictions to time points
        time_points = []
        for i in range(len(risk_scores)):
            time_point = {
                "hour": i,
                "risk_score": float(risk_scores[i]),
                "is_high_risk": bool(risk_scores[i] > 0.5)
            }
            time_points.append(time_point)
        
        # Calculate overall risk
        max_risk = max(risk_scores)
        avg_risk = sum(risk_scores) / len(risk_scores)
        high_risk_hours = sum(1 for score in risk_scores if score > 0.5)
        
        result = {
            "time_points": time_points,
            "summary": {
                "max_risk": float(max_risk),
                "avg_risk": float(avg_risk),
                "high_risk_hours": high_risk_hours,
                "total_hours": len(risk_scores)
            }
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)