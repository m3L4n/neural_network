package main

import (
	"fmt"
	"neural_network/pkg/data"
	"os"

	"github.com/akamensky/argparse"
	"github.com/go-gota/gota/dataframe"
)

func main() {
	parser := argparse.NewParser("print", "Prints provided string to stdout")
	var dataset *os.File = parser.File("d", "dataset", os.O_RDWR, 0600, &argparse.Options{Required: true, Help: "dataset"})
	defer dataset.Close()
	err := parser.Parse(os.Args)
	if err != nil {
		fmt.Print(parser.Usage(err))
		return
	}
	var df dataframe.DataFrame = dataframe.ReadCSV(dataset)
	_, _, errTTS := data.TrainTestSplit(df, 0.25, true)
	if errTTS != nil {
		fmt.Println("Error", errTTS)
		return
	}
}
