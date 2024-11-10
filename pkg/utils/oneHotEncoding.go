package utils

import (
	"errors"
	"fmt"

	"gonum.org/v1/gonum/mat"
)

func OneHotEncoding(y mat.Matrix) (mat.Matrix, error) {
	_, c := y.Dims()
	newMatrix := mat.NewDense(2, c, nil)
	for i := range c {
		label := y.At(0, i)
		if label == 0.0 {
			newMatrix.Set(0, i, 1.0)
			newMatrix.Set(1, i, 0.0)
		} else if label == 1.0 {
			newMatrix.Set(0, i, 0.0)
			newMatrix.Set(1, i, 1.0)
		} else {
			fmt.Println("Error in the labels, value need to be 1.0 or 0.0")
			return nil, errors.New("Error in the labels, value need to be 1.0 or 0.0")
		}
	}
	return newMatrix, nil
}
