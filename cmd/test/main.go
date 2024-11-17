package main

import (
	"fmt"
	"image/color"
	"log"
	ml "neural_network/pkg/ML"
	"os"

	"github.com/akamensky/argparse"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
)



func main() {
	parser := argparse.NewParser("train", "Train the neural network")
	var dataset *os.File = parser.File("d", "dataset", os.O_RDWR, 0600, &argparse.Options{Required: true, Help: "dataset"})
	var epoch *int = parser.Int("e", "epoch", &argparse.Options{Required: false, Help: "number of iteration", Default: 80})
	var learningRate *float64 = parser.Float("l", "learning_rate", &argparse.Options{Required: false, Help: "learning rate", Default: 0.000114})
	var hiddenLayer *[]int = parser.IntList("i", "hidden_layer", &argparse.Options{Required: false, Help: "hidden layer number of neuron for n layer ", Default: []int{24, 24, 24}})
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
	ml.TrainNN(*learningRate, dataset)
}

func PlotData(x [][]float64, path string) {
	p := plot.New()
	xys := make(plotter.XYs, len(x))
	for i := range xys {
		xys[i].X = x[i][0]
		xys[i].Y = x[i][1]
	}
	scatter, _ := plotter.NewScatter(xys)
	scatter.Color = color.RGBA{255, 0, 0, 255}
	scatter.Color = color.RGBA{0, 255, 0, 255}
	p.Add(scatter)
	wt, err := p.WriterTo(300, 300, "png")
	handleError(err)
	f, err := os.Create(path)
	handleError(err)
	defer func() {
		err := f.Close()
		handleError(err)
	}()
	wt.WriteTo(f)
}

func handleError(err error) {
	if err != nil {
		log.Fatal(err)
		fmt.Println(err)
	}
}
