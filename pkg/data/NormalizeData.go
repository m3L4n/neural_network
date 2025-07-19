package data

import (
	"errors"
	"fmt"
	"log"
	"math"

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

func NormTensorByColumn(tensor t.Tensor) t.Tensor {
	if tensor.Dims() != 2 {
		log.Fatalf("Le tensor doit être 2D, mais a %d dimensions", tensor.Dims())
	}

	rows, cols := tensor.Shape()[0], tensor.Shape()[1]
	data := tensor.Data().([]float64)

	normalized := make([]float64, len(data))

	for col := 0; col < cols; col++ {
		min := math.Inf(1)
		max := math.Inf(-1)

		for row := 0; row < rows; row++ {
			val := data[row*cols+col]
			if val < min {
				min = val
			}
			if val > max {
				max = val
			}
		}

		for row := 0; row < rows; row++ {
			idx := row*cols + col
			val := data[idx]
			if max != min {
				normalized[idx] = (val - min) / (max - min)
			} else {
				normalized[idx] = 0.0 
			}
		}
	}

	normTensor := t.New(t.WithShape(rows, cols), t.WithBacking(normalized))
	return normTensor
}
func NormTensor(tensor t.Tensor) t.Tensor {
	// tensorTmp := tensor.Clone().(t.Tensor)
	// minTensor, err := MinTensor(tensorTmp)
	// handleError(err)
	// maxTensor, err := MaxTensor(tensorTmp)
	// handleError(err)
	// normTensor, err := tensorTmp.Apply(func(x float64) float64 {
	// 	return (x - minTensor) / (maxTensor - minTensor)
	// })
	// // fmt.Println(normTensor)
	// handleError(err)
	return NormTensorByColumn(tensor)
}

func handleError(err error) {
	if err != nil {
		log.Fatal(err)
		fmt.Println(err)
	}
}
