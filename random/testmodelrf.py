import os
import pickle
import logging
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_pickle(file_path):
    """
    Load a pickle file from the specified path.
    """
    with open(file_path, 'rb') as file:
        return pickle.load(file)
    
def load_test_dataset(file_path, delimiter, header, lineterminator):
    """
    Load the test dataset from a CSV file.
    """
    logging.info(f"Loading test dataset from: {file_path}")
    return pd.read_csv(file_path, delimiter=delimiter, header=header, lineterminator=lineterminator)
def prepare_test_data(test_dataset, tfidf_vectorizer, label_encoder):
    """
    Prepare test data by combining questions and answers, vectorizing text, and encoding labels.
    """
    questions = test_dataset.iloc[:, 0].values.tolist()
    answers = test_dataset.iloc[:, 1].values.tolist()
    combined_text = [f"{q} {a}" for q, a in zip(questions, answers)]
    
    X_test_tfidf = tfidf_vectorizer.transform(combined_text)
    y_test = label_encoder.transform(answers)
    
    return X_test_tfidf, y_test

def evaluate_model(model, X_test, y_test, label_encoder):
    """
    Evaluate the model's performance on the test set and log the results.
    """
    logging.info("Evaluating model on test data")
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    logging.info(f"Accuracy: {accuracy}")
    logging.info("Classification Report:\n" + report)
    logging.info(f"Confusion Matrix:\n{conf_matrix}")
    
    return accuracy, report, conf_matrix