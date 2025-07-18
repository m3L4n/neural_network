package utils

import t "gorgonia.org/tensor"

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
