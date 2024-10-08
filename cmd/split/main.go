package main

import (
	"fmt"
	"neural_network/pkg/data"
	"os"

	"github.com/akamensky/argparse"
)
func main(){
	parser := argparse.NewParser("print", "Prints provided string to stdout")
	var dataset *os.File = parser.File("d", "dataset", os.O_RDWR, 0600, &argparse.Options{Required: true, Help: "dataset"})
		defer dataset.Close()
			err := parser.Parse(os.Args)
	if err != nil {
		fmt.Print(parser.Usage(err))
		return
	}
	data.Train_test_split(dataset, 0.2)
}