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

## Documentation
Untuk Regexp Queries :
```Go
   primitive.Regex{pattern: "string", options:"i"}
```

sertakan primitve.regex ke dalam bson.M atau bson.D maka string yang dimasukan baik itu lowercase atau UpperCase akan otomatis menyesuaikan sesuai urutan hurufnya .
Pada kasus ini kami menggunakan multiple filter pada fungsi query kami sehingga menjadi:
```
	queries = Stemmer(queries)
	splits := strings.Split(queries, " ")
	if len(splits) >= 5 {
		queries = splits[len(splits)-3] + " " + splits[len(splits)-2] + " " + splits[len(splits)-1]
		filter := bson.M{"questions": primitive.Regex{Pattern: queries, Options: "i"}}
```
 dengan melakukan split dari setiap kata yang ada dan mengambil secara bertahap urutan kata dari setiap kalimat yang dimasukan (full code can be accessed on : kimseokgis/backend-ai/helper/mongo.go)
 
### Why Using regexp?
Keterbatasan GPU dan device mendorong kami menggunakan regex untuk mempermudah ketersediaan chatbot yang kami buat
