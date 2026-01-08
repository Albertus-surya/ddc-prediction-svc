#  Dewey Decimal Classification (DDC) Predictor

A Machine Learning web application designed to classify books into one of the 10 main classes of the **Dewey Decimal Classification (DDC)** system based on their **Title** and **Description**.

## Live Demo
*[(Dewey Decimal Classification (DDC) Predictor)](https://ddc-prediction-svc.streamlit.app/)*

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
