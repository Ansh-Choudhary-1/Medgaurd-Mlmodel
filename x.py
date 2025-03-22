from flask import Flask, jsonify
import random
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

diagnoses = [
    "Atelectasis", "Consolidation", "Infiltration", "Pneumothorax", "Edema",
    "Emphysema", "Fibrosis", "Effusion", "Pneumonia", "Pleural_thickening",
    "Cardiomegaly", "Nodule Mass", "Hernia"
]

@app.route('/random_diagnosis', methods=['GET'])
def get_random_diagnosis():
    random_diagnosis = random.choice(diagnoses)
    return jsonify({"diagnosis": random_diagnosis})

if __name__ == '__main__':
    app.run(debug=True)
