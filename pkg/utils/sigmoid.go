package utils

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

//	func Sigmoid(Z mat.Matrix) mat.Matrix {
//		r, c := Z.Dims()
//		result := mat.NewDense(r, c, nil)
//		result.Apply(func(i, j int, v float64) float64 {
//			return 1.0 / (1.0 + math.Exp(-v))
//		}, Z)
//		return result
//	}
func Sigmoid(Z mat.Matrix) mat.Matrix {
	rows, cols := Z.Dims()
	result := mat.NewDense(rows, cols, nil)

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			// Appliquer la fonction sigmoïde
			value := Z.At(i, j)
			sigmoidedValue := 1.0 / (1.0 + math.Exp(-value))
			result.Set(i, j, sigmoidedValue)
		}
	}
	// r, c := Z.Dims()
	// result := mat.NewDense(r, c, nil)
	// result.Apply(func(i, j int, v float64) float64 {
	// 	return 1.0 / (1.0 + math.Exp(-v))
	// }, Z)
	return result
}
