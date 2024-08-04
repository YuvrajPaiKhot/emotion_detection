from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load the pre-trained model
with open('text_emotion_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the request
    data = request.json
    text = data.get('text', '')

    # Predict the emotion
    prediction = model.predict([text])[0]

    # Return the prediction as a JSON response
    return jsonify({'emotion': prediction})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
