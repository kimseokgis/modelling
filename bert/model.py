# model

import tensorflow as tf
from transformers import TFBertForSequenceClassification, BertTokenizer
import pandas as pd
import csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load the dataset
def filter_valid_rows(row):
    return len(row) == 2
