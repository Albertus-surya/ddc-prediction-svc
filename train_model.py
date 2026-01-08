import pandas as pd
import numpy as np
import re
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score, f1_score
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

print("Loading data...")
df = pd.read_csv("dataset_ready_bolo.csv")
df['deskripsi'] = df['deskripsi'].fillna('')
df['judul_buku'] = df['judul_buku'].fillna('')
df = df.drop_duplicates()

def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

print("Preprocessing...")
df['text'] = (df['judul_buku'] + " " + df['judul_buku'] + " " + df['deskripsi']).apply(preprocess)

X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['main_class'],
    test_size=0.2,
    random_state=42,
    stratify=df['main_class']
)

print("Vectorizing...")
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=12000,
    min_df=2,
    max_df=0.85,
    sublinear_tf=True,
    strip_accents='unicode',
    token_pattern=r'\w{1,}',
    stop_words='english'
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print("Applying SMOTE...")
smote = SMOTE(random_state=42, k_neighbors=5)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_vec, y_train)

print("Training model with GridSearch...")
param_grid = {'C': [0.1, 0.3, 0.5, 0.7, 1.0]}
svc = LinearSVC(max_iter=2500, class_weight='balanced', random_state=42)
grid = GridSearchCV(svc, param_grid, cv=3, scoring='f1_macro', n_jobs=-1)
grid.fit(X_train_balanced, y_train_balanced)

best_model = grid.best_estimator_
y_pred = best_model.predict(X_test_vec)

accuracy = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average='macro')
weighted_f1 = f1_score(y_test, y_pred, average='weighted')

print(f"\nAccuracy: {accuracy:.4f}")
print(f"Macro F1: {macro_f1:.4f}")
print(f"Weighted F1: {weighted_f1:.4f}")
print(f"Best C parameter: {grid.best_params_['C']}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nSaving model and vectorizer...")
with open('ddc_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("Model saved successfully!")
print("Files created: ddc_model.pkl, vectorizer.pkl")