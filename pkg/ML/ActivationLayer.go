package ml

import (
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
