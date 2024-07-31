# modelling
Repo Training Model


### List PreProcessing Inside Repository
1. Tokenizer
 - Tokenizer menggunakan python dengan lib nltk dan pandas
 - Tokenizer menggunakan Golang
2. Stemming
 - Stemming data menggunakan PortStemmer py
 - Stemming menggunakan Sastrawi Untuk bahasa indonesia
 - Stemming Go-Sastrawi Menggunakan golang
 - Normalisasi Imbuhan kata mu (seperti: dirimu menjadi diri kamu)

3. Struktur cleaning folder
```bash
--- cleaning
    --- jaro
    --- bagWords
    --- sastrawi
```

#### Questionable Directory
1. Jaro Winkler
   - Jaro Winkler digunakan untuk mengukur ketepatan atau kesamaan antara kata yang diberikan user dengan kata yang ada pada kalimat dataset
2. Bag Of Words
   - Bag-of-words digunakan untuk menghitung kata dalam kalimat
3. Sastrawi
   - Biasa digunakan untuk melakukan stemming kalimat dalam data (khususnya bahasa indonesia)

### Model yang digunakan
1. Indo-Bert
2. Stacked-LSTM
3. Regexp Queries (golang) (Low Cost Modelling)