package main

// import (
// 	"fmt"
// 	"neural_network/utils"
// 	"os"

// 	"github.com/akamensky/argparse"
// )

// func main() {

// 	parser := argparse.NewParser("print", "Prints provided string to stdout")
// 	// var loss *string = parser.Selector("o", "loss", []string{"binaryCrossentropy", "binaryCrossentropy1", "binaryCrossentropy2"}, &argparse.Options{Required: false, Help: "loss of the neural network", Default: "binaryCrossentropy"})
// 	var action *string = parser.Selector("a", "action", []string{"split", "train", "test"}, &argparse.Options{Required: true, Help: "action to perform"})
// 	var dataset *os.File = parser.File("d", "dataset", os.O_RDWR, 0600, &argparse.Options{Required: true, Help: "dataset"})
// 	// var batch *int = parser.Int("b", "batch", &argparse.Options{Required: false, Help: "size of number of data perform per epoch", Default: 24})
// 	// var epoch *int = parser.Int("e", "epoch", &argparse.Options{Required: false, Help: "number of iteration", Default: 80})
// 	// var learning_rate *float64 = parser.Float("l", "learning_rate", &argparse.Options{Required: false, Help: "learning rate", Default: 0.0314})
// 	// var hidden_layer *[]int = parser.IntList("h", "hidden_layer", &argparse.Options{Required: false, Help: "hidden layer number of neuron for n layer", Default: []int{24, 24, 24}})
// 	defer dataset.Close()
// 	err := parser.Parse(os.Args)
// 	if err != nil {
// 		fmt.Print(parser.Usage(err))
// 		return
// 	}
// 	switch *action {
// 	case "split":
// 		utils.Train_test_split(dataset, 0.2)
// 		return
// 	case "train":
// 		fmt.Println("need to train")
// 		return
// 	case "test":
// 		fmt.Println("need to train")
// 		return
// 	}

// }
