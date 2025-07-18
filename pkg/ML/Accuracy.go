package ml

import (
	"neural_network/pkg/utils"

	t "gorgonia.org/tensor"
)

func Accuracy(yPred, yTrue t.Tensor) float64 {

	newYPred := utils.Argmax(yPred)
	for idx := 0; idx < yTrue.Shape()[0]; idx++ {
	}
	// fmt.Println(newYPred.Shape()[0], yTrue.Shape()[0])
	predictionSum := 0
	for idx := 0; idx < yTrue.Shape()[0]; idx++ {
		classPred, err := newYPred.At(idx, 0)
		handleError(err)
		classTrue, err := yTrue.At(idx, 0)
		handleError(err)
		if classPred.(float64) == classTrue.(float64) {
			predictionSum += 1
		} else {
			predictionSum += 0
		}
	}
	var meanAccuracy float64 = float64(predictionSum) / float64(yTrue.Shape()[0])
	return meanAccuracy
}
