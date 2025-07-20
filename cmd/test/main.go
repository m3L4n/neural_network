package main

import (
	"fmt"
	ml "neural_network/pkg/ML"
	"neural_network/pkg/utils"
	"os"

	"github.com/akamensky/argparse"
)

func main() {
	parser := argparse.NewParser("train", "Train the neural network")
	var dataset *os.File = parser.File("d", "dataset", os.O_RDWR, 0600, &argparse.Options{Required: true, Help: "dataset"})
	defer dataset.Close()
	err := parser.Parse(os.Args)
	if err != nil {
		fmt.Print(parser.Usage(err))
		return
	}

	x, y, _, _ := utils.PreprocessData(dataset, false)
	nn := ml.LoadNeuralNetwork("./model/model.bin")
	nn.Predict(x, y)

}
