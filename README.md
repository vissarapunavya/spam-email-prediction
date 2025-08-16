# 📧 Spam Email Predictor  

##  Overview  
This project detects whether an email message is **Spam** or **Ham (Not Spam)** using a **Logistic Regression model** trained on a text dataset.  
The model is integrated with a **Flask web application** and deployed online.  

 **Live Demo:** [Try it here](https://spam-email-predictor-w7i4.onrender.com)  
 **GitHub Repo:** [View on GitHub](https://github.com/vissarapunavya/spam-email-prediction)  
---

## Project Structure  
- `data/` → Raw and cleaned data files  
- `src/` → Preprocessing, training, and prediction scripts  
- `templates/` → HTML files for Flask web app  
- `app.py` → Flask application script  
- `requirements.txt` → Required Python libraries  

---

##  How to Run Locally  

1. Clone the repository:  
   ```bash
   git clone https://github.com/vissarapunavya/spam-email-prediction.git
   cd spam-email-prediction
2.Install the dependencies
   pip install -r requirements.txt
3.run the web app
   python app.py
4.Open your browser and go to
  http://127.0.0.1:5000/

