from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

app = Flask(__name__)
CORS(app)

# Load the model
model = joblib.load('text_emotion_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data['text']
    
    # Preprocess the text (if needed)
    # prediction = model.predict([preprocessed_text])
    prediction = model.predict([text])
    
    return jsonify({'emotion': prediction[0]})

if __name__ == '__main__':
    app.run(debug=False)
