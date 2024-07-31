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

# Baca file CSV ke dalam DataFrame
df = pd.read_csv('dataset/dataset.csv')

# Asumsikan kolom kalimat bernama 'sentence'
df['stemmed_sentence'] = df['sentence'].apply(stem_sentence)

# Tampilkan DataFrame dengan kalimat yang sudah di-stem
print(df.head())

# Simpan DataFrame hasil stemming ke file CSV baru
df.to_csv('dataset/stemmed_dataset.csv', index=False)