import joblib

def load_artifacts():
    model = joblib.load("spam_model.joblib")
    vectorizer = joblib.load("vectorizer.joblib")
    return model, vectorizer

def predict_spam(text, model, vectorizer):
    features = vectorizer.transform([text])
    pred = model.predict(features)
    return "Spam" if pred[0] == 1 else "Ham"

if __name__ == "__main__":
    model, vectorizer = load_artifacts()
    user_input = input("Enter your email/message: ")
    print(predict_spam(user_input, model, vectorizer))
