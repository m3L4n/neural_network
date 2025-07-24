package ml

import (
	"errors"
	"math"

	t "gorgonia.org/tensor"
)

type ActivationReLu struct {
	Output t.Tensor
	Input  t.Tensor
	DInput t.Tensor
}


func NewActivation() *ActivationReLu {
	return &ActivationReLu{Output: t.New(t.Of(t.Float64))}
}

// Forward method applies the ReLU activation function to the input tensor
//
// It replaces negative values with zero and keeps positive values unchanged.
// It is used to introduce non-linearity into the network, which allows it to learn complex patterns.
// formula : ReLU(x) = max(0, x)
func (a *ActivationReLu) Forward(inputs t.Tensor) {
	a.Input = inputs
	zeros := t.New(t.WithShape(inputs.Shape()...), t.Of(t.Float64))
	output, err := t.MaxBetween(zeros, inputs)
	handleError(err)

	a.Output = output
}

//Backward method computes the gradient of the ReLU activation function
// It sets the gradient to zero for negative inputs and keeps it unchanged for positive inputs.
func (a *ActivationReLu) Backward(dvalues t.Tensor) {
	output := dvalues.Clone().(t.Tensor)
	shapeDvalues := dvalues.Shape()
	for i := 0; i < shapeDvalues[0]; i++ {

		for j := 0; j < shapeDvalues[1]; j++ {
			valueInput, err := a.Input.At(i, j)
			handleError(err)
			valueInputF := valueInput.(float64)
			if valueInputF <= 0 {
				output.SetAt(0.0, i, j)
			}
		}

	}
	a.DInput = output
}

type ActivationSoftmax struct {
	Outpout t.Tensor
	DInput  t.Tensor
}

func NewActivationSoftmax() *ActivationSoftmax {
	return &ActivationSoftmax{Outpout: t.New(t.Of(t.Float64))}
}

// Forward method applies the softmax activation function to the input tensor
// It computes the exponential of each element, normalizes by the sum of exponentials, and
// stores the result in the Outpout tensor.
// It will give us the probability for each classes to be the real classes of the input tensor.
// formula : softmax(x_i) = exp(x_i) / sum(exp(x_j))
func (s *ActivationSoftmax) Forward(layerOutput t.Tensor) {
	shape := layerOutput.Shape()
	expValues := t.New(t.WithShape(shape...), t.Of(t.Float64))

	// to find the max in the tensor for each row
	for i := range shape[0] {
		max, err := layerOutput.At(i, 0)
		handleError(err)
		for j := range shape[1] {
			value, err := layerOutput.At(i, j)
			handleError(err)

			if max.(float64) < value.(float64) {
				max = value
			}
		}
	// softmax application (only exp for each value)
		for j := range shape[1] {
			value, err := layerOutput.At(i, j)
			handleError(err)
			res := math.Exp(value.(float64) - max.(float64))
			expValues.SetAt(res, i, j)
		}
	}
	sum, err := t.Sum(expValues, 1)
	sum.Reshape(sum.Shape()[0], 1)
	handleError(err)
	prob := t.New(t.WithShape(shape...), t.Of(t.Float64))
	shapeExp := expValues.Shape()
	// softmax final application  ( e^x_i -max  / sum(e^x_i) )
	for i := range shapeExp[0] {
		for j := range shapeExp[1] {
			value, err := expValues.At(i, j)
			handleError(err)
			sumValue, err := sum.At(i, 0)
			prob.SetAt(value.(float64)/sumValue.(float64), i, j)
		}
	}
	s.Outpout = prob

}

// Backward method computes the gradient of the softmax activation function
// It adjusts the gradients based on the predicted and true labels.
// It will give us the gradient of the softmax activation function.
// formula = gradient[i,trueClass]= probability[i,trueClass] − 1
func (s *ActivationSoftmax) Backward(dvalues, y t.Tensor) error {
	shapeY := y.Shape()
	shapeDValues := dvalues.Shape()
	output := dvalues.Clone().(t.Tensor)
	if shapeY[0] != shapeDValues[0] {
		return errors.New("Error shape 1 of y and dvalues  need to be the same")
	}
	for i := 0; i < shapeY[0]; i++ {
		valueY, err := y.At(i, 0)
		handleError(err)
		value, err := dvalues.At(i, int(valueY.(float64)))
		output.SetAt((value.(float64) - 1.), i, int(valueY.(float64)))

	}
	normFactor := float64(shapeY[0])
	outputNormTmp := output.Clone().(t.Tensor)
	// normalize the output by the number of elements to avoid exploding gradients
	outputNorm, err := outputNormTmp.Apply(func(x float64) float64 {
		return x / normFactor
	})
	handleError(err)

	s.DInput = outputNorm
	return nil

}
