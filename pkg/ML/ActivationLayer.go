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

func NewActivation() ActivationReLu {
	return ActivationReLu{Output: t.New(t.Of(t.Float64))}
}

func (a *ActivationReLu) Forward(inputs t.Tensor) {
	a.Input = inputs
	zeros := t.New(t.WithShape(inputs.Shape()...), t.Of(t.Float64))
	output, err := t.MaxBetween(zeros, inputs)
	handleError(err)

	a.Output = output
}

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

func NewActivationSoftmax() ActivationSoftmax {
	return ActivationSoftmax{Outpout: t.New(t.Of(t.Float64))}
}

func (s *ActivationSoftmax) Forward(layerOutput t.Tensor) {
	shape := layerOutput.Shape()
	expValues := t.New(t.WithShape(shape...), t.Of(t.Float64))
	for i := range shape[0] {
		max, err := layerOutput.At(i, 0)
		handleError(err)
		for j := range shape[1] {
			value, err := layerOutput.At(i, j)
			handleError(err)

			// 	value = layerOutput[i][j]
			if max.(float64) < value.(float64) {
				max = value
			}
		}

		for j := range shape[1] {
			value, err := layerOutput.At(i, j)
			handleError(err)
			res := math.Pow(math.E, (value.(float64) - max.(float64)))
			expValues.SetAt(res, i, j)
		}
	}
	sum, err := t.Sum(expValues, 1)
	sum.Reshape(sum.Shape()[0], 1)
	handleError(err)
	prob := t.New(t.WithShape(shape...), t.Of(t.Float64))
	shapeExp := expValues.Shape()
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

func (s *ActivationSoftmax) Backward(dvalues, y t.Tensor) error {
	shapeY := y.Shape()
	shapeDValues := dvalues.Shape()

	outpout := dvalues.Clone().(t.Tensor)
	if shapeY[0] != shapeDValues[0] {
		return errors.New("Error shape 1 of y and dvalues  need to be the same")
	}
	for i := 0; i < shapeY[0]; i++ {
		valueY, err := y.At(i, 0)
		handleError(err)
		value, err := dvalues.At(i, int(valueY.(float64)))
		outpout.SetAt((value.(float64) - 1.), i, int(valueY.(float64)))

	}
	normFactor := float64(shapeY[0])
	ouputNormTmp := outpout.Clone().(t.Tensor)
	ouputNorm, err := ouputNormTmp.Apply(func(x float64) float64 {
		return x / normFactor
	})
	handleError(err)

	s.DInput = ouputNorm
	return nil

}
