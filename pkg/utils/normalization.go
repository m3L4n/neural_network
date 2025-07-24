package utils

import (
	"math"

	t "gorgonia.org/tensor"
)

func NormalizeTensorZScore(tensor t.Tensor) t.Tensor {
	shape := tensor.Shape()
	rows, cols := shape[0], shape[1]
	data := tensor.Data().([]float64)
	normData := make([]float64, len(data))

	for col := 0; col < cols; col++ {
		sum := 0.0
		for row := 0; row < rows; row++ {
			sum += data[row*cols+col]
		}
		mean := sum / float64(rows)

		variance := 0.0
		for row := 0; row < rows; row++ {
			diff := data[row*cols+col] - mean
			variance += diff * diff
		}
		stddev := math.Sqrt(variance / float64(rows))

		for row := 0; row < rows; row++ {
			normData[row*cols+col] = (data[row*cols+col] - mean) / stddev
		}
	}

	normTensor := t.New(t.WithShape(rows, cols), t.WithBacking(normData))
	return normTensor
}
