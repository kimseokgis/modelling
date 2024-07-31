package bagWords

import "strings"

func BagOfWords(text string) map[string]int {
	// Membuat map untuk menyimpan frekuensi kata
	wordFreq := make(map[string]int)

	// Memisahkan teks menjadi kata-kata
	words := strings.Fields(text)

	// Menghitung frekuensi setiap kata
	for _, word := range words {
		wordFreq[strings.ToLower(word)]++ // Menggunakan toLower untuk konsistensi
	}

	return wordFreq
}
