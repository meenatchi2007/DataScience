import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Sample movie reviews data (positive and negative) - expanded for better balance
reviews = [
    # Positive reviews
    "This movie was fantastic! I loved every minute of it.",
    "An absolute masterpiece, highly recommend to everyone.",
    "Great acting and storyline, couldn't put it down.",
    "The best film I've seen this year, amazing plot.",
    "Incredible special effects and plot twists, thrilling.",
    "Wonderful characters and beautiful cinematography.",
    "Hilarious comedy that had me laughing throughout.",
    "Touching story that moved me to tears.",
    "Excellent direction and score, a true gem.",
    "Captivating from start to finish, must-watch.",
    # Negative reviews
    "This film was terrible, complete waste of time.",
    "Boring and predictable, don't bother watching it.",
    "Poor acting and awful script, very disappointing.",
    "One of the worst movies I've ever seen.",
    "Disappointing from start to finish, no redeeming qualities.",
    "Confusing plot and bad pacing, couldn't finish it.",
    "Overrated and dull, not worth the hype.",
    "Weak story and flat characters, avoid it.",
    "Technical issues and poor editing, frustrating.",
    "Lame ending and unconvincing performances."
]

labels = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # 1 for positive, 0 for negative

# Create a pipeline with TF-IDF vectorizer and Logistic Regression
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=1000, stop_words='english')),
    ('clf', LogisticRegression(random_state=42))
])

# Train the model
pipeline.fit(reviews, labels)

# Save the trained model to model.pkl
joblib.dump(pipeline, 'model.pkl')

print("Model trained and saved to model.pkl")
