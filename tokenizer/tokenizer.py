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

# Membersihkan dan melakukan tokenisasi pada kolom question dan answer
dataset['cleaned_question'] = dataset['question'].apply(clean_text)
dataset['cleaned_answer'] = dataset['answer'].apply(clean_text)

dataset['tokenized_question'] = dataset['cleaned_question'].apply(word_tokenize)
dataset['tokenized_answer'] = dataset['cleaned_answer'].apply(word_tokenize)