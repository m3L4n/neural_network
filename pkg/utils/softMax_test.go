package utils

import (
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestSoftmax(t *testing.T) {

	zeroVector := mat.NewVecDense(4, []float64{0, 0, 0, 0})
	expectedZero := mat.NewVecDense(4, []float64{0.25, 0.25, 0.25, 0.25})
	result := softmax(zeroVector)
	if !mat.EqualApprox(result, expectedZero, 1e-10) {
		t.Errorf("Softmax vector of zero get : obtenu\n%v\nexpected\n%v", mat.Formatted(result), mat.Formatted(expectedZero))
	}

	positiveVector := mat.NewVecDense(4, []float64{1, 2, 3, 4})
	expectedPositive := mat.NewVecDense(4, []float64{
		0.0320586033,
		0.0871443187,
		0.2368828181,
		0.6439142599,
	})
	result = softmax(positiveVector)
	if !mat.EqualApprox(result, expectedPositive, 1e-10) {
		t.Errorf("Softmax vector of positive float : get\n%v\nexpected\n%v", mat.Formatted(result), mat.Formatted(expectedPositive))
	}

	negativeVector := mat.NewVecDense(4, []float64{-1, -2, -3, -4})
	expectedNegative := mat.NewVecDense(4, []float64{
		0.6439142599,
		0.2368828181,
		0.0871443187,
		0.0320586033,
	})
	result = softmax(negativeVector)
	if !mat.EqualApprox(result, expectedNegative, 1e-10) {
		t.Errorf("Softmax vector of negative float : get\n%v\n expected \n%v", mat.Formatted(result), mat.Formatted(expectedNegative))
	}

	mixedVector := mat.NewVecDense(4, []float64{1, -1, 2, 0})
	expectedMixed := mat.NewVecDense(4, []float64{
		0.2368828181,
		0.0320586033,
		0.6439142599,
		0.0871443187,
	})
	result = softmax(mixedVector)
	if !mat.EqualApprox(result, expectedMixed, 1e-10) {
		t.Errorf("Softmax vector mixt value : get \n%v\n expected\n%v", mat.Formatted(result), mat.Formatted(expectedMixed))
	}
}
