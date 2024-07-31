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

# Drop kolom 'id'
dataset = dataset.drop(columns=['_id/$oid'])

# Definisikan simbol-simbol yang ingin diganti dan penggantinya
replace_dict = {
    '@': '',
    '#': '',
    '&': 'and',
    '!': '',
    '"': '',
    "'": '',
    ",": '',
    '.': '',
    '\n': ' ',
    '\n\n': ''
}

# Fungsi untuk mengganti simbol pada string
def replace_symbols(text, replace_dict):
    for symbol, replacement in replace_dict.items():
        text = text.replace(symbol, replacement)
    return text

dataset['questions'] = dataset['questions'].apply(lambda x: replace_symbols(str(x), replace_dict))
dataset['answer'] = dataset['answer'].apply(lambda x: replace_symbols(str(x), replace_dict))

# Membersihkan dan melakukan tokenisasi pada kolom question dan answer
dataset['cleaned_question'] = dataset['questions'].apply(clean_text)
dataset['cleaned_answer'] = dataset['answer'].apply(clean_text)

dataset['tokenized_question'] = dataset['cleaned_question'].apply(word_tokenize)
dataset['tokenized_answer'] = dataset['cleaned_answer'].apply(word_tokenize)

# Membagi dataset menjadi set pelatihan dan pengujian
train, test = train_test_split(dataset, test_size=0.2, random_state=42)

# Menampilkan hasil tokenisasi dan pembagian dataset
print("Dataset Pelatihan (Train):")
print(train[['question', 'tokenized_question', 'answer', 'tokenized_answer']])

print("\nDataset Pengujian (Test):")
print(test[['question', 'tokenized_question', 'answer', 'tokenized_answer']])