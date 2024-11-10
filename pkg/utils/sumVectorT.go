package utils

import "gonum.org/v1/gonum/mat"

func SumVectorT(vectorT mat.Matrix) (mat.Dense, error) {
	r, _ := vectorT.Dims()
	result := mat.NewDense(r, 1, nil)
	denseMatrix := mat.DenseCopyOf(vectorT)
	for i := range r {
		row := denseMatrix.RowView(i)
		sum := mat.Sum(row)
		result.Set(i, 0, sum)
	}

	return *result, nil
}
