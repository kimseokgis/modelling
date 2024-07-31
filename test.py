import tensorflow as tf
from transformers import TFBertForSequenceClassification, BertTokenizer
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import csv

# Load the tokenizer and model
tokenizer = BertTokenizer.from_pretrained('indobert_model')
model = TFBertForSequenceClassification.from_pretrained('indobert_model')

