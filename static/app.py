from flask import Flask, request, jsonify, send_file
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('model.pkl')

@app.route('/')
def home():
    return send_file('index.html')

@app.route('/style.css')
def css():
    return send_file('style.css')

@app.route('/predict', methods=['POST'])
def predict():
    review = request.form.get('review')
    if not review:
        return jsonify({'error': 'No review provided'}), 400

    # Predict sentiment
    prediction = model.predict([review])[0]
    sentiment = 'positive' if prediction == 1 else 'negative'

    return jsonify({'sentiment': sentiment})

if __name__ == '__main__':
    app.run(debug=True)
