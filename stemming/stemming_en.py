import pandas as pd
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Pastikan untuk mengunduh data yang diperlukan
nltk.download('punkt')

# Inisialisasi PorterStemmer
porter = PorterStemmer()

# Fungsi untuk stemming menggunakan PorterStemmer
def stem_sentence(sentence):
    words = word_tokenize(sentence)
    return " ".join([porter.stem(word) for word in words])