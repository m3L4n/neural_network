package ml

import (
	"errors"
	"math"

	t "gorgonia.org/tensor"
)

func BinaryCrossEntropy(yPred, y t.Tensor) (float64, error) {
	epsilon := 1e-7
	shapeY := y.Shape()
	shapeYPred := yPred.Shape()

	if shapeY[0] != shapeYPred[0] {
		return 0, errors.New("shape of prediction and labels must be the same")
	}

	var bce = 0.0

	for i := 0; i < shapeY[0]; i++ {
		valueLabel, err := y.At(i, 0)
		if err != nil {
			return 0, err
		}
		valueLabelF := valueLabel.(float64)

		valuePrediction, err := yPred.At(i, 1)

		if err != nil {
			return 0, err
		}
		valuePredictionF := valuePrediction.(float64)
		bce += valueLabelF*math.Log(math.Max(valuePredictionF, epsilon)) + (1-valueLabelF)*math.Log(math.Max(1-valuePredictionF, 1-epsilon))
	}

	return -bce / float64(shapeY[0]), nil
}
