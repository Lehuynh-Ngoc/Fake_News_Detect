import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
import os
import sys

# Add parent dir to path to import src
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.preprocess import clean_text

from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import FunctionTransformer
import numpy as np

def get_text_features(text_series):
    # Tính toán các đặc trưng bổ sung: độ dài, số lượng dấu chấm than, số lượng chữ số
    features = []
    for text in text_series:
        text_str = str(text)
        length = len(text_str)
        exclamation_count = text_str.count('!')
        digit_count = sum(c.isdigit() for c in text_str)
        capitals_count = sum(1 for c in text_str if c.isupper())
        features.append([length, exclamation_count, digit_count, capitals_count])
    return np.array(features)

def train_model(data_path, model_output_path):
    print(f"Loading data from {data_path}...")
    if not os.path.exists(data_path):
        print(f"Error: File {data_path} not found.")
        return

    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    text_col = 'text' if 'text' in df.columns else 'post_message'
    df.dropna(subset=[text_col, 'label'], inplace=True)
    
    print("Preprocessing and balancing data...")
    df['clean_text'] = df[text_col].apply(clean_text)

    # Cân bằng dữ liệu (Oversampling)
    df_majority = df[df.label == 0]
    df_minority = df[df.label == 1]
    df_minority_upsampled = resample(df_minority, replace=True, n_samples=len(df_majority), random_state=42)
    df_balanced = pd.concat([df_majority, df_minority_upsampled])

    X = df_balanced['clean_text']
    y = df_balanced['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

    print("Training Advanced Hybrid Model...")
    
    # Kết hợp Word TF-IDF và Char TF-IDF (Phân tích sâu cấu trúc từ như Huffman)
    features = FeatureUnion([
        ('word_tfidf', TfidfVectorizer(ngram_range=(1, 3), sublinear_tf=True, max_features=5000)),
        ('char_tfidf', TfidfVectorizer(ngram_range=(2, 5), analyzer='char', sublinear_tf=True, max_features=5000)),
    ])

    pipeline = Pipeline([
        ('features', features),
        ('clf', RandomForestClassifier(n_estimators=300, max_depth=None, min_samples_split=2, random_state=42, n_jobs=-1))
    ])

    pipeline.fit(X_train, y_train)

    print("Evaluating model...")
    y_pred = pipeline.predict(X_test)
    print(classification_report(y_test, y_pred))
    
    print(f"Saving model to {model_output_path}...")
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(pipeline, model_output_path)
    print("Training complete!")

if __name__ == "__main__":
    DATA_PATH = os.path.join("data", "fake_news.csv")
    MODEL_PATH = os.path.join("models", "fake_news_model.pkl")
    train_model(DATA_PATH, MODEL_PATH)