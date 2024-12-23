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
	var epoch *int = parser.Int("e", "epoch", &argparse.Options{Required: false, Help: "number of iteration", Default: 1000})
	var learningRate *float64 = parser.Float("l", "learning_rate", &argparse.Options{Required: false, Help: "learning rate", Default: 0.030})
	var batch *int = parser.Int("b", "batch", &argparse.Options{Required: false, Help: "size of number of data perform per epoch", Default: 100})
	var hiddenLayer *[]int = parser.IntList("i", "hidden_layer", &argparse.Options{Required: false, Help: "hidden layer number of neuron for n layer ", Default: []int{50, 32, 64}})
	defer dataset.Close()
	err := parser.Parse(os.Args)
	if err != nil {
		fmt.Print(parser.Usage(err))
		return
	}
	if *epoch <= 0 || *learningRate <= 0 || len(*hiddenLayer) == 0 {
		fmt.Println("Error need to receives only postiv number ")
		os.Exit(1)
	}
	ml.TrainNN(*learningRate, dataset, *epoch, *hiddenLayer, *batch)
}
