import pandas as pd
import nltk
import re
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split

# Pastikan untuk mendownload paket yang dibutuhkan
nltk.download('punkt')

# Fungsi untuk membersihkan teks menggunakan regex
def clean_text(text):
    # Menghapus karakter non-alfabet
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Mengubah teks menjadi huruf kecil
    text = text.lower()
    return text

# Membaca dataset dari file CSV
file_path = 'dataset/output.csv'
dataset = pd.read_csv(file_path)