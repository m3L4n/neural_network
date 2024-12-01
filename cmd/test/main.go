package main

import (
	"fmt"
	ml "neural_network/pkg/ML"
	"os"

	"github.com/akamensky/argparse"
)

func main() {
	parser := argparse.NewParser("train", "Train the neural network")
	var dataset *os.File = parser.File("d", "dataset", os.O_RDWR, 0600, &argparse.Options{Required: true, Help: "dataset"})
	// var batch *int = parser.Int("b", "batch", &argparse.Options{Required: false, Help: "size of number of data perform per epoch", Default: 100})
	defer dataset.Close()
	err := parser.Parse(os.Args)
	if err != nil {
		fmt.Print(parser.Usage(err))
		return
	}
	ml.PredictNN(dataset)
}
