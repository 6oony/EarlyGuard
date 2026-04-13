# EarlyGuard

Early-stage ransomware detection using system-call analysis and machine
learning.

---

## Overview

EarlyGuard is a machine learning-based project that focuses on detecting
ransomware behaviour at an early stage using system-call logs.

Instead of detecting ransomware after damage is done, this system tries
to identify suspicious behaviour from the initial execution patterns of
a program.

The system takes system-call sequences as input and classifies them as
either benign or ransomware.

---

## Objective

The main objective of this project is to explore whether early-stage
system-call activity can be used to detect ransomware before encryption
or file damage begins.

---

## Methodology

1.  Data Preparation\
    System-call data is generated in CSV format using a synthetic data
    generator.\
    The dataset includes both benign and ransomware-like sequences with
    overlap and noise.

2.  Feature Extraction\
    The sequences are converted into text format and processed using
    TF-IDF with n-grams (1 to 3).

3.  Model Training\
    Multiple machine learning models are used:

- Logistic Regression\
- Naive Bayes\
- Support Vector Machine (SVM)\
- Random Forest

4.  Model Evaluation\
    Models are evaluated using accuracy, precision, recall, and
    F1-score.\
    Cross-validation is also used for better reliability.

5.  Prediction\
    Only the first 120 system calls are used to simulate early-stage
    detection.\
    The model predicts whether the behaviour is benign or ransomware and
    provides a confidence score.

---

## Features

- Single file analysis\
- Batch analysis\
- Model comparison\
- Confidence score and risk indication\
- Downloadable results\
- Streamlit-based interface

---

## Project Structure

EarlyGuard/ │ ├── app.py\
├── train_model.py\
├── generate_sample_data.py\
│ ├── dataset/\
├── models/

---

## How to Run

Step 1 (optional -- generate data):\
python generate_sample_data.py

Step 2 (train models):\
python train_model.py

Step 3 (run the app):\
streamlit run app.py

---

## Limitations

This project works on pre-generated system-call logs and does not
include real-time system monitoring.

The dataset used is synthetic, so results may differ in real-world
scenarios.

---

## Future Scope

- Integration with real-time system-call monitoring\
- Use of real malware datasets\
- Deployment as a full detection system

---

## Conclusion

This project shows that ransomware behaviour can be identified using
early-stage system-call patterns with machine learning.

---

## Author

Your Name
