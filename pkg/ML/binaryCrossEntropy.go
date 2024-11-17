package ml

import (
	"errors"
	"math"

	t "gorgonia.org/tensor"
)

func BinaryCrossEntropy(yPred, y t.Tensor) (float64, error) {
	epsilon := 1e-10
	newYPred := ArgmaxPred(yPred)
	shapeY := y.Shape()
	shapeYPred := newYPred.Shape()
	if shapeY[0] != shapeYPred[0] {
		return 0, errors.New("shape of prediction and labels have to be the same")
	}
	var bce = 0.0
	for i := 0; i < shapeY[0]; i++ {
		valueLabel, err := y.At(i, 0)
		valueLabelF := valueLabel.(float64)
		handleError(err)
		valuePrediction, err := newYPred.At(i, 0)
		valuePredictionF := valuePrediction.(float64)
		handleError(err)
		// bce += valueLabelF*math.Log(math.Max(valuePredictionF, epsilon)) + (1-valueLabelF)*math.Log(math.Max(1-valuePredictionF, epsilon))
		bce += valueLabelF*math.Log(math.Max(valuePredictionF, epsilon)) + (1-valueLabelF)*math.Log(math.Max(1-valuePredictionF, epsilon))
	}

	return -bce / float64(shapeY[0]), nil
}
