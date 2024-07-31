package sastrawi

import (
	"fmt"
	"testing"
)

func TestStemmer(t *testing.T) {
	str := "aku mau pergi ke bali dulu yah"
	newSentence := Stemmer(str)
	fmt.Println(newSentence)
}
