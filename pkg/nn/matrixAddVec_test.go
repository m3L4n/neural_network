package nn

import (
	"errors"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestMatrixAddVec(t *testing.T) {
	// Test with a zero matrix
	zeroMatrix := mat.NewDense(2, 3, []float64{
		0, 0, 0,
		0, 0, 0,
	})
	vector := mat.NewVecDense(2, []float64{1, 2}) // Length should match the number of rows
	expected := mat.NewDense(2, 3, []float64{
		1, 1, 1,
		2, 2, 2,
	})

	result, err := MatrixAddVec(zeroMatrix, vector)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	if !mat.EqualApprox(result, expected, 1e-10) {
		t.Errorf("Adding a vector to a zero matrix failed: got\n%v\nexpected\n%v", mat.Formatted(result), mat.Formatted(expected))
	}

	// Test with a matrix containing positive values
	positiveMatrix := mat.NewDense(2, 3, []float64{
		1, 2, 3,
		4, 5, 6,
	})
	vector = mat.NewVecDense(2, []float64{1, 1}) // Change the vector length to match matrix rows
	expected = mat.NewDense(2, 3, []float64{
		2, 3, 4,
		5, 6, 7,
	})

	result, err = MatrixAddVec(positiveMatrix, vector)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	if !mat.EqualApprox(result, expected, 1e-10) {
		t.Errorf("Adding a vector to a positive matrix failed: got\n%v\nexpected\n%v", mat.Formatted(result), mat.Formatted(expected))
	}

	// Test with a vector length mismatch
	mismatchVector := mat.NewVecDense(4, []float64{1, 1, 1, 1}) // Length does not match the matrix
	_, err = MatrixAddVec(positiveMatrix, mismatchVector)
	if err == nil {
		t.Fatalf("Expected an error for vector length mismatch, but got none")
	}
	if !errors.Is(err, ErrVectorLengthMismatch) {
		t.Fatalf("Unexpected error: %v", err)
	}

	// Test with a matrix containing negative values
	negativeMatrix := mat.NewDense(2, 3, []float64{
		-1, -2, -3,
		-4, -5, -6,
	})
	vector = mat.NewVecDense(2, []float64{1, 2}) // Change vector length to match the matrix
	expected = mat.NewDense(2, 3, []float64{
		0, -1, -2,
		-2, -3, -4,
	})

	result, err = MatrixAddVec(negativeMatrix, vector)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	if !mat.EqualApprox(result, expected, 1e-10) {
		t.Errorf("Adding a vector to a negative matrix failed: got\n%v\nexpected\n%v", mat.Formatted(result), mat.Formatted(expected))
	}

	// Test with a matrix containing mixed values
	mixedMatrix := mat.NewDense(2, 3, []float64{
		1, -1, 2,
		-2, 2, -3,
	})
	vector = mat.NewVecDense(2, []float64{2, 2}) // Change vector length to match the matrix
	expected = mat.NewDense(2, 3, []float64{
		3, 1, 4,
		0, 4, -1,
	})

	result, err = MatrixAddVec(mixedMatrix, vector)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	if !mat.EqualApprox(result, expected, 1e-10) {
		t.Errorf("Adding a vector to a mixed matrix failed: got\n%v\nexpected\n%v", mat.Formatted(result), mat.Formatted(expected))
	}
}
