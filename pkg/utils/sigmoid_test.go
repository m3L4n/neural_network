package utils

// func matApproxEqual(a, b mat.Matrix, tol float64) bool {
// 	r, c := a.Dims()
// 	rb, cb := b.Dims()
// 	if r != rb || c != cb {
// 		return false
// 	}
// 	for i := 0; i < r; i++ {
// 		for j := 0; j < c; j++ {
// 			if math.Abs(a.At(i, j)-b.At(i, j)) > tol {
// 				return false
// 			}
// 		}
// 	}
// 	return true
// }

// func TestSigmoid(t *testing.T) {
// 	matrixData := []float64{0.0, 2.0, -2.0, 4.0}
// 	Z := mat.NewDense(2, 2, matrixData)

// 	expected := mat.NewDense(2, 2, []float64{
// 		0.5, 0.88079708, 0.11920292, 0.98201379,
// 	})

// 	result := Sigmoid(Z)

// 	if !matApproxEqual(result, expected, 1e-6) {
// 		t.Errorf("Error sigmoid : get %v, expected %v", mat.Formatted(result), mat.Formatted(expected))
// 	}

// 	matrixData2 := []float64{100.0, -100.0, 0.0, 1.0}
// 	Z2 := mat.NewDense(2, 2, matrixData2)

// 	expected2 := mat.NewDense(2, 2, []float64{
// 		1.0, 0.0, 0.5, 0.73105858,
// 	})

// 	result2 := Sigmoid(Z2)

// 	if !matApproxEqual(result2, expected2, 1e-6) {
// 		t.Errorf("Error sigmoid  : get %v, expected %v", mat.Formatted(result2), mat.Formatted(expected2))
// 	}

// 	matrixData3 := []float64{0.0, 0.0, 0.0, 0.0}
// 	Z3 := mat.NewDense(2, 2, matrixData3)

// 	expected3 := mat.NewDense(2, 2, []float64{
// 		0.5, 0.5, 0.5, 0.5,
// 	})

// 	result3 := sigmoid(Z3)

// 	if !matApproxEqual(result3, expected3, 1e-6) {
// 		t.Errorf("Error sigmoid (zero matrix ) : get %v, expected %v", mat.Formatted(result3), mat.Formatted(expected3))
// 	}
// }
