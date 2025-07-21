package main

import (
	"fmt"
	"math/rand"
	ml "neural_network/pkg/ML"
	"os"
	"neural_network/pkg/utils"
	"github.com/akamensky/argparse"
)

func main() {
	parser := argparse.NewParser("train", "Train the neural network")
	var dataset *os.File = parser.File("d", "dataset", os.O_RDWR, 0600, &argparse.Options{Required: true, Help: "dataset"})
	var epoch *int = parser.Int("e", "epoch", &argparse.Options{Required: false, Help: "number of iteration", Default: 1200})
	var learningRate *float64 = parser.Float("l", "learning_rate", &argparse.Options{Required: false, Help: "learning rate", Default: 0.012})
	var hiddenLayer *[]int = parser.IntList("i", "hidden_layer", &argparse.Options{Required: false, Help: "hidden layer number of neuron for n layer ", Default: []int{ 32, 16}})
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
	rand.Seed(12) // Seed the random number generator for reproducibility
	xTrainTensor, yTraintensor, xTest, yTest := utils.PreprocessData(dataset, true)
	neuralNetwork := ml.NewNeuralNetwork(*learningRate, xTrainTensor, *hiddenLayer, *epoch)
	neuralNetwork.Fit(xTrainTensor, yTraintensor, xTest, yTest)
	ml.SaveNeuralNetwork(neuralNetwork)
}
