package utils

import (
	"errors"

	"gonum.org/v1/gonum/mat"
)

var ErrVectorLengthMismatch = errors.New("matrix rows must match vector length")

func MatrixAddVec(matrix mat.Matrix, vector mat.Vector) (mat.Matrix, error) {
	rowM, columnM := matrix.Dims()
	if rowM != vector.Len() {
		return nil, ErrVectorLengthMismatch
	}
	vRep := mat.NewDense(rowM, columnM, nil)
	var result mat.Dense

	for i := 0; i < vRep.RawMatrix().Cols; i++ {

		vRep.SetCol(i, mat.Col(nil, 0, vector))
	}
	result.Add(matrix, vRep)
	return &result, nil
}
