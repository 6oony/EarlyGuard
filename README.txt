# EarlyGuard  
Early-stage ransomware detection using system-call analysis and machine learning  

---

## Overview  
EarlyGuard is a machine learning-based project that focuses on detecting ransomware behaviour at an early stage using system-call logs.  
The idea behind this project is that instead of detecting ransomware after damage is done, it is possible to identify suspicious behaviour from the initial execution patterns of a program.

The system takes system-call sequences as input and classifies them as either benign or ransomware based on learned patterns.

---

## Objective  
The main objective of this project is to explore whether early-stage system-call activity can be used to detect ransomware before encryption or file damage begins.

---

## Methodology  

### 1. Data Preparation  
System-call data is generated in CSV format using a synthetic data generator.  
The dataset includes both benign and ransomware-like sequences with overlap and noise to simulate realistic behaviour.

### 2. Feature Extraction  
The system-call sequences are converted into text format and processed using TF-IDF vectorization with n-grams (1 to 3).  
This helps capture patterns and relationships between system calls.

### 3. Model Training  
Multiple machine learning models are trained and evaluated, including:
- Logistic Regression  
- Naive Bayes  
- Support Vector Machine (SVM)  
- Random Forest  

### 4. Model Evaluation  
Models are evaluated using:
- Accuracy  
- Precision  
- Recall  
- F1-score  

Cross-validation is also used to ensure consistent performance.

### 5. Prediction  
During testing, only the early portion of system calls (first 120 calls) is used to simulate early detection.  
The model then predicts whether the behaviour is benign or ransomware and provides a confidence score.

---

## Features  
- Analysis of single system-call log files  
- Batch processing of multiple files  
- Model comparison using evaluation metrics  
- Confidence score and risk indication  
- Downloadable results  
- Interactive interface using Streamlit  

---

## Project Structure  
EarlyGuard/
│
├── app.py # Streamlit interface
├── train_model.py # Model training and evaluation
├── generate_sample_data.py # Synthetic dataset generation
│
├── dataset/ # Generated system-call logs
├── models/ # Saved models and vectorizer


---

## How to Run  

Step 1 (optional – generate sample data):  
python generate_sample_data.py  

Step 2 (train models):  
python train_model.py  

Step 3 (run the application):  
streamlit run app.py  

---

## Limitations  
This project currently works on pre-generated system-call logs and does not include real-time system monitoring.  
The dataset used is synthetic, so results may differ in real-world scenarios.

---

## Future Scope  
- Integration with real-time system-call monitoring tools  
- Use of real malware datasets  
- Deployment as an endpoint detection system  

---

## Conclusion  
This project demonstrates that ransomware behaviour can be identified from early-stage system-call patterns using machine learning techniques.  
It highlights the possibility of proactive detection instead of reactive response.

---

## Author  
Boon Suni  
