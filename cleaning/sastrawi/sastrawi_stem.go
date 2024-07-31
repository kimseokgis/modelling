package sastrawi

import (
	"github.com/RadhiFadlillah/go-sastrawi"
	"regexp"
	"strings"
)

func Stemmer(Sentences string) (newString string) {
	dictionary := sastrawi.DefaultDictionary()
	stemmer := sastrawi.NewStemmer(dictionary)
	for _, word := range sastrawi.Tokenize(Sentences) {
		//fmt.Println(word)
		newString = newString + " " + stemmer.Stem(word)
		//fmt.Println(newString)
	}
	return strings.TrimSpace(newString)
}

func SeparateSuffixMu(word string) string {
	// Regex untuk mendeteksi kata dengan imbuhan "mu" di akhir
	re := regexp.MustCompile(`(\w+)(mu)$`)

	// Cek apakah kata cocok dengan regex
	if re.MatchString(word) {
		// Ganti "mu" dengan " kamu"
		return re.ReplaceAllString(word, "$1 kamu")
	}

	// Jika tidak ada imbuhan "mu", kembalikan kata asli
	return word
}
