package ml

import (
	t "gorgonia.org/tensor"
)

func Argmax(yPred t.Tensor) t.Tensor {
	shapeYPred := yPred.Shape()
	newYPred := t.New(t.WithShape(yPred.Shape()[0], 1), t.Of(t.Float64))
	for i := 0; i < shapeYPred[0]; i++ {
		indexMax := 0
		valueMax, err := yPred.At(i, indexMax)
		handleError(err)
		for j := 0; j < shapeYPred[1]; j++ {
			valueTmp, err := yPred.At(i, j)
			handleError(err)
			if (valueMax.(float64)) < valueTmp.(float64) {
				valueMax = valueTmp
				indexMax = j
			}
		}
		newYPred.SetAt(float64(indexMax), i, 0)
	}
	return newYPred
}

func Accuracy(yPred, yTrue t.Tensor) float64 {

	newYPred := Argmax(yPred)
	for idx := 0; idx < yTrue.Shape()[0]; idx++ {
	}
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
