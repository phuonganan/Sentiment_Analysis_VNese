from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load model
MODEL_PATH = "models/svm.pkl"
model = pickle.load(open(MODEL_PATH, 'rb'))

# Load label encoder
LABEL_ENCODER_PATH = "models/label_encoder.pkl"
label_encoder = pickle.load(open(LABEL_ENCODER_PATH, 'rb'))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data['text']

    # Predict sentiment
    label = model.predict([text])
    sentiment = label_encoder.inverse_transform(label)

    return jsonify({'sentiment': sentiment[0]})

if __name__ == '__main__':
    app.run(debug=True)