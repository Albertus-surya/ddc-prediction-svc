# 📚 Dewey Decimal Classification (DDC) Predictor

![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn)

A Machine Learning web application designed to classify books into one of the 10 main classes of the **Dewey Decimal Classification (DDC)** system based on their **Title** and **Description**.

## Live Demo
*(Paste link aplikasi Streamlit Anda di sini setelah deploy)*

## Model & Methodology

This project solves a multi-class text classification problem using **Linear Support Vector Classification (Linear SVC)**.

### Key Techniques Used:
1.  **Text Preprocessing**:
    * Regex cleaning (removing special characters).
    * **Feature Engineering**: Concatenating `Title + Title + Description` to give more weight to the book title.
2.  **Handling Imbalanced Data**:
    * Applied **SMOTE (Synthetic Minority Over-sampling Technique)** with `k_neighbors=5` to balance the dataset before training.
3.  **Vectorization**:
    * **TF-IDF** with n-grams (1,2) and sublinear classification.
4.  **Model Optimization**:
    * Hyperparameter tuning using **GridSearchCV** to find the best `C` parameter.
    * Trained with `class_weight='balanced'`.

### Performance
Based on the test set (20% split):
* **Accuracy**: ~68.77%
* **Weighted F1-Score**: ~0.69
* **Best Parameters**: LinearSVC with optimized C value.

## Project Structure

```bash
├── app.py                  # Main Streamlit application script
├── train_model.py          # Script for training, SMOTE, and GridSearch
├── ddc_model.pkl           # Trained LinearSVC Model
├── vectorizer.pkl          # Fitted TF-IDF Vectorizer
├── dataset_ready_bolo.csv  # Dataset used for training
├── requirements.txt        # Python dependencies
└── README.md               # Documentation