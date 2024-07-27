import os
import pandas as pd
import pickle
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_output_directory(path):
    """
    Create the output directory jika tidak ditemukan
    """
    logging.info(f"Creating output directory at: {path}")
    os.makedirs(path, exist_ok=True)

def load_dataset(file_path, delimiter, header, lineterminator):
    """
    Load the dataset from a CSV file.
    """
        logging.info(f"Loading dataset from: {file_path}")
    dataset = pd.read_csv(file_path, delimiter=delimiter, header=header, lineterminator=lineterminator)

    
combined_text = [q + " " + a for q, a in zip(questions, answers)]

label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(answers)

# Split the data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(combined_text, labels, test_size=0.2, random_state=42)

tfidf_vectorizer = TfidfVectorizer(max_features=1000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train_tfidf, y_train)

y_pred = rf_classifier.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)

print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(report)   

import pickle

with open(os.path.join(path, 'rf_classifier.pkl'), 'wb') as model_file:
    pickle.dump(rf_classifier, model_file)

with open(os.path.join(path, 'tfidf_vectorizer.pkl'), 'wb') as vectorizer_file:
    pickle.dump(tfidf_vectorizer, vectorizer_file)

with open(os.path.join(path, 'label_encoder.pkl'), 'wb') as encoder_file:
    pickle.dump(label_encoder, encoder_file)