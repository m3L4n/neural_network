package utils

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

func Softmax(matrix mat.Matrix) mat.Matrix {

	r, c := matrix.Dims()
	newMatrix := mat.NewDense(r, c, nil)
	for i := 0; i < c; i++ {
		var sum = 0.0
		for row := 0; row < r; row++ {
			sum += math.Exp(newMatrix.At(row, i))
		}
		for row := range r {
			value := newMatrix.At(row, i)
			softmaxValue := math.Exp(value) / sum
			newMatrix.Set(row, i, softmaxValue)
		}
	}
	return newMatrix
}

// func Softmax(matrix *mat.Dense) *mat.Dense {
// 	var sum float64
// 	// Calculate the sum

// 	for _, v := range matrix.RawMatrix().Data {
// 		fmt.Println("in softmax", v)
// 		sum += math.Exp(v)
// 	}

// 	resultMatrix := mat.NewDense(matrix.RawMatrix().Rows, matrix.RawMatrix().Cols, nil)
// 	// Calculate softmax value for each element
// 	resultMatrix.Apply(func(i int, j int, v float64) float64 {
// 		return math.Exp(v) / sum
// 	}, matrix)

// 	return resultMatrix
// }

// func Softmax(matrix mat.Matrix) mat.Matrix {
// 		fmt.Println("Hello in Softmax", matrix)
// 	// Format(matrix)
// 	// Obtenir les dimensions de la matrice
// 	rows, cols := matrix.Dims()

// 	// Créer une nouvelle matrice pour stocker le résultat
// 	result := mat.NewDense(rows, cols, nil)

// 	for i := 0; i < rows; i++ {
// 		var max float64
// 		// Trouver le maximum de la ligne pour la stabilité numérique
// 		row := mat.Row(nil, i, matrix)
// 		max = row[0]
// 		for _, value := range row {
// 			if value > max {
// 				max = value
// 			}
// 		}

// 		// Calculer les exponentielles et la somme des exponentielles
// 		var sum float64
// 		for j := 0; j < cols; j++ {
// 			expVal := math.Exp(row[j] - max) // Soustraction de max pour la stabilité
// 			result.Set(i, j, expVal)
// 			sum += expVal
// 		}

// 		// Normaliser pour obtenir la softmax
// 		for j := 0; j < cols; j++ {
// 			result.Set(i, j, result.At(i, j)/sum)
// 		}
// 	}

// 	return result
// }

func softmax(Z mat.Vector) mat.Matrix {
	length := Z.Len()
	result := mat.NewVecDense(length, nil)
	prob := mat.NewVecDense(length, nil)
	sum := 0.0
	for i := 0; i < length; i++ {
		tmpExp := math.Exp(Z.AtVec(i))
		sum += tmpExp
		result.SetVec(i, tmpExp)
	}
	for j := 0; j < length; j++ {
		value := result.AtVec(j) / sum
		prob.SetVec(j, value)
	}
	return prob
}
