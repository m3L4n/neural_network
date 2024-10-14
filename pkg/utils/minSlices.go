package utils

// MinSlice return the min float in a slice of float64
// return 0 if len of slice  == 0
func MinSlice(slice []float64) float64 {
	var minFloat = 0.
	if len(slice) > 0 {

		minFloat = slice[0]
		if minFloat < 0 {

			minFloat = minFloat - (minFloat * 2)
		}
		for _, value := range slice {
			if value < 0 {
				value = value - (value * 2)
			}
			if minFloat > value {
				minFloat = value
			}
		}
	}
	return minFloat

}

// MinMatrix return the min float in a Matrix of float64
// return 0 if len of matrix  == 0
func MinMatrix(matrix [][]float64) float64 {
	if len(matrix) == 0 {
		return 0
	}
	var minFloat = 0.
	if len(matrix[0]) > 0 {

		minFloat = MinSlice(matrix[0])
		for _, row := range matrix {
			if len(row) > 0 {
				minTmp := MinSlice(row)
				if minTmp < minFloat {
					minFloat = minTmp
				}
			}
		}
	}
	return minFloat
}
