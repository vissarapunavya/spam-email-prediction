import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load clean data
df = pd.read_csv("../data/clean_mail_data.csv")

X = df['Message']
y = df['Category']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Vectorize text
vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
X_train_features = vectorizer.fit_transform(X_train)
X_test_features = vectorizer.transform(X_test)

# Train model
model = LogisticRegression()
model.fit(X_train_features, y_train)

# Evaluate the model
y_pred = model.predict(X_test_features)
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

# Save model and vectorizer with joblib
joblib.dump(model, "spam_model.joblib")
joblib.dump(vectorizer, "vectorizer.joblib")
