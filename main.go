package main

import (
	"fmt"
	"github.com/kimseokgis/modelling/cleaning"
)

func main() {
	// Memanggil fungsi Cleaningdata
	err := cleaning.Cleaningdata()
	if err != nil {
		fmt.Printf("Error saat membersihkan data: %v\n", err)
		return
	}

	fmt.Println("Data telah dibersihkan dengan sukses!")
}
