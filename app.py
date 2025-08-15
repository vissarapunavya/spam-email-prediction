from flask import Flask, request, render_template
import joblib

app = Flask(__name__)

# Load model and vectorizer once when the app starts
model = joblib.load("src/spam_model.joblib")
vectorizer = joblib.load("src/vectorizer.joblib")

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    if request.method == "POST":
        user_text = request.form["message"]
        features = vectorizer.transform([user_text])
        pred = model.predict(features)
        result = "Spam" if pred[0] == 1 else "ham"
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
