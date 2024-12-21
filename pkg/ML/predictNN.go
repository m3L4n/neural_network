package ml

import (
	"neural_network/pkg/utils"
	"os"
)

func PredictNN(dataset *os.File) {
	x, y, _, _ := utils.PreprocessData(dataset, false)
	nn := LoadNeuralNetwork("./model/model.bin")
	nn.Predict(x, y)


}
