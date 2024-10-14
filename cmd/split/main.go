package main

import (
	"fmt"
	"neural_network/pkg/data"
	"os"

	"github.com/akamensky/argparse"
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
	_, _, errTTS := data.TrainTestSplit(dataset, 0.2)
	if errTTS != nil {
		fmt.Println("Error", errTTS)
		return
	}
}
