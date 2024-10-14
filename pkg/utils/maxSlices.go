package utils

// MaxSlice return the max float in a slice of float64
func MaxSlice(slice []float64) float64 {
	var maxFloat = 0.
	if len(slice) > 0 {

		maxFloat = slice[0]
		if maxFloat < 0 {

			maxFloat = maxFloat - (maxFloat * 2)
		}
		for _, value := range slice {
			if value < 0 {
				value = value - (value * 2)
			}
			if maxFloat < value {
				maxFloat = value
			}
		}
	}
	return maxFloat

}

// MaxMatrix return the max float in a Matrix of float64
func MaxMatrix(matrix [][]float64) float64 {
	if len(matrix) == 0 {
		return 0
	}
	var maxFloat = 0.
	if len(matrix[0]) > 0 {

		maxFloat = MaxSlice(matrix[0])
		for _, row := range matrix {
			if len(row) > 0 {
				maxTmp := MaxSlice(row)
				if maxTmp > maxFloat {
					maxFloat = maxTmp
				}
			}
		}
	}
	return maxFloat
}
