package utils

import (
	"errors"
	"fmt"
	"log"

	t "gorgonia.org/tensor"
)

func MaxTensor(tensor t.Tensor) (float64, error) {
	shape := tensor.Shape()
	if shape[0] == 0 || shape[1] == 0 {
		return 0.0, errors.New("tensor need to have a valid len")
	}
	max, err := tensor.At(0, 0)
	handleError(err)
	maxF := max.(float64)
	for i := 0; i < shape[0]; i++ {
		for j := 0; j < shape[1]; j++ {
			valueTensor, err := tensor.At(i, j)
			handleError(err)
			valueTensorF := valueTensor.(float64)
			if valueTensorF > maxF {
				maxF = valueTensorF
			}
		}
	}
	return maxF, nil
}

func handleError(err error) {
	if err != nil {
		log.Fatal(err)
		fmt.Println(err)
	}
}