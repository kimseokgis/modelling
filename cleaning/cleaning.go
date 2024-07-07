package cleaning

import (
	"context"
	"fmt"
	"strings"
)

func Cleaningdata() (Err error) {
	// Membuat konteks dengan timeout
	//ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	//defer cancel()
	db := DBAdmn("AI")
	var datas []Dataset
	re := 100
	for i := 1; i < re; i++ {
		params := PaginationParams{
			Page:  int64(i),
			Limit: 150,
			//Offset: 0,
		}
		data, err := GetAllDatasets(context.TODO(), params, db)
		if err != nil {
			return err
		}
		datas = data

		fmt.Printf("%d\n", i)

		fmt.Println(datas)

		for _, v := range datas {
			for _, symbol := range symbols {
				if strings.ContainsAny(v.Answers, symbol) {
					v.Answers = strings.Replace(v.Answers, symbol, "", 10)
				}
			}
			valueUpdated := Dataset{
				Questions: v.Questions,
				Answers:   v.Answers,
			}

			err := UpdateDatasetByID(v.ID, valueUpdated, db)
			if err != nil {
				return err
			}
		}
		defer db.Client().Disconnect(context.TODO())
	}
	return
}

var symbols = []string{
	"!",
	"@",
	"#",
	"$",
	"%",
	"^",
	"&",
	"*",
	"(",
	")",
	"-",
	"+",
	"=",
	"[",
	"]",
	"{",
	"}",
	";",
	":",
	"'",
	//"\"",
	//"\\",
	"|",
	",",
	".",
	"<",
	">",
	"/",
	"?",
	"\n",
	",",
}
