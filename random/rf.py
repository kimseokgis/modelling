import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

path = "output_dir_14k/"
os.makedirs(path, exist_ok=True)

dataset = pd.read_csv('data.csv', delimiter="|", header=None, lineterminator='\n')
if dataset.shape[1] < 2:
    raise ValueError("The dataset does not have the expected number of columns.")

questions = dataset.iloc[:, 0].values.tolist()
answers = dataset.iloc[:, 1].values.tolist()

combined_text = [q + " " + a for q, a in zip(questions, answers)]

label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(answers)

# Split the data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(combined_text, labels, test_size=0.2, random_state=42)