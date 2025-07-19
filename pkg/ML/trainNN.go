package ml

import (
	"image/color"
	"log"
	"neural_network/pkg/utils"
	"os"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
)

func TrainNN(learningRate float64, dataset *os.File, epoch int, hiddenLayer []int) {
	xTrainTensor, yTraintensor, xTest, yTest := utils.PreprocessData(dataset, true)
	neuralNetwork := NewNeuralNetwork(learningRate, xTrainTensor, hiddenLayer, epoch)
	neuralNetwork.Fit(xTrainTensor, yTraintensor, xTest, yTest)
	SaveNeuralNetwork(neuralNetwork)
}
func PlotData(path string, train []float64, test []float64) {
	p := plot.New()
	xys := make(plotter.XYs, len(train))
	for i := range xys {

		xys[i].Y = train[i]
		xys[i].X = float64(i)
	}
	filled, err := plotter.NewLine(xys)
	if err != nil {
		log.Panic(err)
	}
	p.Add(filled)
	xysTest := make(plotter.XYs, len(test))
	for i := range xysTest {

		xysTest[i].Y = test[i]
		xysTest[i].X = float64(i)
	}
	filledTest, err := plotter.NewLine(xysTest)
	filledTest.Color = color.RGBA{R: 255, A: 255, B: 255}
	p.Add(filledTest)
	err = p.Save(1024, 1024, path)
	if err != nil {
		log.Panic(err)
	}
}
