from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os

app = Flask(__name__)
CORS(app)

try:
    # Load the model
    model = joblib.load('text_emotion_model.pkl')
except Exception as e:
    print(f"Error loading model: {e}")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        text = data['text']
        
        # Preprocess the text (if needed)
        prediction = model.predict([text])
        
        # Convert numpy.int64 to int
        emotion = int(prediction[0])
        
        return jsonify({'emotion': emotion})
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
