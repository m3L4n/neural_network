package utils

import "gonum.org/v1/gonum/mat"

func SubMatrixScalar(scalar float64, matrix mat.Matrix) mat.Dense {
	r, c := matrix.Dims()
	var sliceScalar = make([]float64, r*c)
	for i := 0; i < r*c; i++ {
		sliceScalar[i] = scalar
	}
	matrixScalar := mat.NewDense(r, c, sliceScalar)
	var result mat.Dense
	result.Sub(matrixScalar, matrix)
	return result
}
