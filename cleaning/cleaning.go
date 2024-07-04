package cleaning

import (
	"context"
	"strings"
	"time"
)

func Cleaningdata() (Err error) {
	// Membuat konteks dengan timeout
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	data, err := GetAllDatasets(ctx)
	if err != nil {
		return err
	}

	for _, v := range data {
		symbols := []string{
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

		for _, symbol := range symbols {
			if strings.ContainsAny(v.Answers, symbol) {
				v.Answers = strings.Replace(v.Answers, symbol, "", 10)
			}
		}

		valueUpdated := Dataset{
			//ID:        primitive.ObjectID{},
			Questions: v.Questions,
			Answers:   v.Answers,
		}

		err = UpdateDatasetByID(v.ID, valueUpdated)
		if err != nil {
			return err
		}
	}
	return err
}
