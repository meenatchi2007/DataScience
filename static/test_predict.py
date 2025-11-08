import joblib

# Load the model
model = joblib.load('static/model.pkl')

# Test positive review
positive_review = "This movie was fantastic! I loved every minute of it."
prediction = model.predict([positive_review])[0]
sentiment = 'positive' if prediction == 1 else 'negative'
print(f"Prediction for positive review: {sentiment}")

# Test negative review
negative_review = "This film was terrible, waste of time."
prediction = model.predict([negative_review])[0]
sentiment = 'positive' if prediction == 1 else 'negative'
print(f"Prediction for negative review: {sentiment}")
