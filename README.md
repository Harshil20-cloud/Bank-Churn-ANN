# 🏦 Bank Customer Churn Prediction using Artificial Neural Network (ANN)

## 📌 Project Overview
This project predicts whether a bank customer will churn (leave the bank) or stay using an Artificial Neural Network (ANN).  
A Streamlit web app is used for easy user interaction and real-time predictions.

---

## 🧠 Model Details
- Model Type: Artificial Neural Network (ANN)
- Problem Type: Binary Classification
- Libraries Used: TensorFlow, Keras, Scikit-learn, Pandas, NumPy, Streamlit

---

## 📂 Project Files
- model.py → Data preprocessing and ANN model building  
- main.py → Model training and evaluation  
- app.py → Streamlit web application  
- dataset.csv → Input dataset  
- model.h5 → Saved trained model  
- scaler.pkl → Feature scaler  
- requirements.txt → Required libraries  

---

## ⚙️ Workflow
1. Load dataset  
2. Data preprocessing (missing values, encoding, scaling)  
3. Split data into training and testing  
4. Build ANN model  
5. Train model  
6. Evaluate model  
7. Save model  
8. Run Streamlit app for prediction  

---

## 🚀 How to Run

### Install dependencies
pip install -r requirements.txt

### Train model
python main.py

### Run Streamlit app
streamlit run app.py

---

## 🎯 Output
- 1 → Customer will churn  
- 0 → Customer will not churn  

---

## 📊 Features Used
- Credit Score  
- Geography  
- Gender  
- Age  
- Balance  
- Number of Products  
- Has Credit Card  
- Is Active Member  
- Estimated Salary  

---

## 👨‍💻 Author
Harshil Gautam