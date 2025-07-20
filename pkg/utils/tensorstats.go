package utils

import (
	"errors"
	"fmt"
	"log"
	"math"

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

func MeanTensor(tensor t.Tensor) (float64, error) {
	sum, err := t.Sum(tensor, 0)
	if err != nil {
		return 0, err
	}

	shape := tensor.Shape()
	numElements := shape[0] * shape[1]
	if len(shape) == 1 {
		numElements = shape[0]
	}

	mean := sum.Data().([]float64)[0] / float64(numElements)
	return mean, nil
}

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

func StdTensor(tensor t.Tensor) (float64, error) {
	mean, err := MeanTensor(tensor)
	if err != nil {
		return 0, err
	}

	squaredDiff := tensor.Clone().(t.Tensor)
	squaredDiff, _ = squaredDiff.Apply(func(x float64) float64 {
		return math.Pow(x-mean, 2)
	})
	if err != nil {
		return 0, err
	}

	variance, err := t.Sum(squaredDiff, 0)
	if err != nil {
		return 0, err
	}

	shape := tensor.Shape()
	numElements := shape[0] * shape[1]
	if len(shape) == 1 {
	}

	varianceValue := variance.Data().([]float64)[0] / float64(numElements)
	stddev := math.Sqrt(varianceValue)

	return stddev, nil
}
func handleError(err error) {
	if err != nil {
		log.Fatal(err)
		fmt.Println(err)
	}
}
