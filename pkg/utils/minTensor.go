package utils

import (
	"errors"

	t "gorgonia.org/tensor"
)

func MinTensor(tensor t.Tensor) (float64, error) {
	shape := tensor.Shape()
	if shape[0] == 0 || shape[1] == 0 {
		return 0.0, errors.New("tensor need to have a valid len")
	}
	min, err := tensor.At(0, 0)
	handleError(err)
	minF := min.(float64)
	for i := 0; i < shape[0]; i++ {
		for j := 0; j < shape[1]; j++ {
			valueTensor, err := tensor.At(i, j)
			handleError(err)
			valueTensorF := valueTensor.(float64)
			if valueTensorF < minF {
				minF = valueTensorF
			}
		}
	}
	return minF, nil
}
