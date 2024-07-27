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