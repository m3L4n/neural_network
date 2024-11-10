package utils

// func TestSumVectorT(t *testing.T) {
// 	data1 := []float64{1, 2, 3, 4}
// 	vectorT1 := mat.NewDense(1, 4, data1)

// 	result, err := SumVectorT(vectorT1)
// 	if err != nil {
// 		t.Fatalf("Unexpected error for row vector: %v", err)
// 	}

// 	expected1 := mat.NewDense(1, 1, []float64{10})
// 	if !mat.Equal(result, expected1) {
// 		t.Errorf("Error: expected result %v, got %v", expected1, result)
// 	}

// 	data2 := []float64{5, 6, 7}
// 	vectorT2 := mat.NewDense(3, 1, data2)

// 	result, err = SumVectorT(vectorT2)
// 	if err != nil {
// 		t.Fatalf("Unexpected error for column vector: %v", err)
// 	}

// 	expected2 := mat.NewDense(3, 1, []float64{5, 6, 7})
// 	if !mat.Equal(&result, expected2) {
// 		t.Errorf("Error: expected result %v, got %v", expected2, result)
// 	}

// }
