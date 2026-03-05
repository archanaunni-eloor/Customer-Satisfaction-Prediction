Customer Support Satisfaction Prediction
This project uses Machine Learning and Natural Language Processing (NLP) to predict customer satisfaction ratings based on support ticket data. It features a complete pipeline from data cleaning and feature engineering to a live web dashboard built with Streamlit.
## 🚀 Live Demo
Check out the live web application here: https://customer-satisfaction-prediction-878swvs9wobpnvepmkrxeb.streamlit.app/


🚀 Project Overview
The goal of this project is to help support teams predict how satisfied a customer is likely to be based on their ticket details, such as age, ticket priority, and the time taken to resolve the issue.

Key Features:
Predictive Modeling: Uses a Random Forest Classifier to predict CSAT (Customer Satisfaction) scores.

NLP Integration: Experimental Sentiment Analysis on ticket descriptions using TextBlob.

Interactive Dashboard: A Streamlit web app for real-time predictions.

Automated Preprocessing: Handles One-Hot Encoding and feature scaling automatically.

📈 Performance & Observations
During development, two versions of the model were tested. Interestingly, adding NLP features caused a slight drop in accuracy, likely due to the synthetic nature of the text data.
Model Version,Accuracy
Baseline (Numerical & Categorical Features),0.2238 (Selected)
NLP Enhanced (Sentiment Analysis),0.1931
Note: The decrease in accuracy suggests that the ticket descriptions provided in the dataset contain "noise" or repetitive patterns that do not strongly correlate with the actual ratings.
Tech Stack
Language: Python 3.x

Libraries: Pandas, NumPy, Scikit-learn, Joblib, TextBlob

Deployment: Streamlit Cloud

Version Control: Git & GitHub
File Structure
app.py: The Streamlit web application code.

model.pkl: The trained Random Forest model.

model_columns.pkl: The exact list of 57 features used during training (required for feature alignment).

requirements.txt: List of dependencies for deployment.

customer_support_tickets.csv: The raw dataset.
How It Works (Feature Alignment)
To ensure the app doesn't crash, we use Feature Alignment. Since the model was trained on 57 features (due to One-Hot Encoding), the app collects user input, encodes it, and then uses .reindex() to fill missing categories with zeros. This ensures the input shape always matches the model's requirements.

✨ Future Improvements
Implement more advanced NLP models like BERT for better sentiment understanding.

Collect real-world (non-synthetic) data to improve accuracy.

Integrate the model into a Telegram Bot for automated support monitoring.
