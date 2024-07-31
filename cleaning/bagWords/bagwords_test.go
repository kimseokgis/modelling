package bagWords

import (
	"fmt"
	"testing"
)

func TestBagWords(t *testing.T) {
	text := "Hello world! Hello Golang. Welcome to the world of Golang."

	// Mendapatkan representasi Bag-of-Words
	bow := BagOfWords(text)

	// Mencetak hasilnya
	for word, freq := range bow {
		fmt.Printf("%s: %d\n", word, freq)
	}

}
