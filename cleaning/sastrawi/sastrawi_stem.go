package sastrawi

import (
	"github.com/RadhiFadlillah/go-sastrawi"
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
