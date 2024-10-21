package main

import (
	"fmt"
	"neural_network/pkg/data"
	"os"

	"github.com/akamensky/argparse"
)

func main() {
	fmt.Println("Hello from the train program")
	parser := argparse.NewParser("train", "Train the neural network")
	var dataset *os.File = parser.File("d", "dataset", os.O_RDWR, 0600, &argparse.Options{Required: true, Help: "dataset"})
	var batch *int = parser.Int("b", "batch", &argparse.Options{Required: false, Help: "size of number of data perform per epoch", Default: 24})
	var epoch *int = parser.Int("e", "epoch", &argparse.Options{Required: false, Help: "number of iteration", Default: 80})
	var learningRate *float64 = parser.Float("l", "learning_rate", &argparse.Options{Required: false, Help: "learning rate", Default: 0.0314})
	var hiddenLayer *[]int = parser.IntList("i", "hidden_layer", &argparse.Options{Required: false, Help: "hidden layer number of neuron for n layer ", Default: []int{24, 24, 24}})
	// var loss *string = parser.Selector("o", "loss", []string{"binaryCrossentropy", "binaryCrossentropy1", "binaryCrossentropy2"}, &argparse.Options{Required: false, Help: "loss of the neural network", Default: "binaryCrossentropy"})
	defer dataset.Close()
	err := parser.Parse(os.Args)
	if err != nil {
		fmt.Print(parser.Usage(err))
		return
	}
	if *batch <= 0 || *epoch <= 0 || *learningRate <= 0 || len(*hiddenLayer) == 0 {
		fmt.Println("Error need to receives only postiv number ")
		os.Exit(1)
	}
	xTrain, yTrain, xTest, yTest, err := data.ProcessData(dataset)
	if err != nil {
		fmt.Println("Error in cmd train", err)
		return
	}
	rowXTrain, columnXtrain := xTrain.Dims()
	rowXTest, columnXtest := xTest.Dims()
	fmt.Printf("x train shape ( %v , %v )\n", rowXTrain, columnXtrain)
	fmt.Printf("x validation  shape ( %v , %v )\n", rowXTest, columnXtest)
	fmt.Println(yTrain, yTest)
	// neuralNetwork := nn.NeuralNetwork{}
	// neuralNetworkPtr, err := neuralNetwork.CreateNetwork(*batch, *hiddenLayer, *epoch, *learningRate, data)
	//
	//	if err != nil {
	//		fmt.Println(err)
	//		os.Exit(1)
	//	}
	//
	// neuralNetworkPtr.Fit(data, labels)
}
