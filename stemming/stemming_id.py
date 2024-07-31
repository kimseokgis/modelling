import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.tokenize import word_tokenize

# Inisialisasi stemmer bahasa Indonesia
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Fungsi untuk stemming menggunakan Sastrawi
def stem_sentence(sentence):
    words = word_tokenize(sentence)
    return " ".join([stemmer.stem(word) for word in words])

# Baca file CSV ke dalam DataFrame
df = pd.read_csv('dataset/dataset.csv')

# Asumsikan kolom kalimat bernama 'sentence'
df['stemmed_sentence'] = df['question'].apply(stem_sentence)

# Tampilkan DataFrame dengan kalimat yang sudah di-stem
print(df.head())

# Simpan DataFrame hasil stemming ke file CSV baru
df.to_csv('dataset/id/stemmed_dataset.csv', index=False)