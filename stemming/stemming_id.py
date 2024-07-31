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