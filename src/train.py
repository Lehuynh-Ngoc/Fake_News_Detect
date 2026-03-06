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

    # Check columns
    text_col = 'text' if 'text' in df.columns else 'post_message'
    if text_col not in df.columns or 'label' not in df.columns:
        print(f"Error: CSV must contain '{text_col}' and 'label' columns. Found: {df.columns.tolist()}")
        return

    # Drop NaNs
    df.dropna(subset=[text_col, 'label'], inplace=True)
    
    print(f"Preprocessing text from column '{text_col}'...")
    # Use only first 10k rows for faster training if dataset is too large, 
    # but 4.5MB (~4000 rows) is fine for full training.
    df['clean_text'] = df[text_col].apply(clean_text)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['clean_text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
    )

    print("Training model...")
    # Pipeline: TF-IDF -> Logistic Regression
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2))),
        ('clf', LogisticRegression(random_state=42))
    ])

    pipeline.fit(X_train, y_train)

    print("Evaluating model...")
    y_pred = pipeline.predict(X_test)
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    print(f"Saving model to {model_output_path}...")
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(pipeline, model_output_path)
    print("Training complete!")

if __name__ == "__main__":
    DATA_PATH = os.path.join("data", "fake_news.csv")
    MODEL_PATH = os.path.join("models", "fake_news_model.pkl")
    train_model(DATA_PATH, MODEL_PATH)