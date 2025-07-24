package ml

import (
	t "gorgonia.org/tensor"
)

type OptimizerGd struct {
	LearningRate float64
}

func NewOptimizerGd(learningRate float64) *OptimizerGd {
	return &OptimizerGd{LearningRate: learningRate}
}

func (o *OptimizerGd) UpdateParameter(weight, dweight, bias, dbias t.Tensor) (t.Tensor, t.Tensor) {
	newWeight := dweight.Clone().(t.Tensor)
	newWeightModified, err := newWeight.Apply(func(x float64) float64 {
		res := -o.LearningRate * x
		return res
	})
	handleError(err)
	UpdatedWeight, err := t.Add(weight, newWeightModified)
	handleError(err)

	biasCopy := dbias.Clone().(t.Tensor)
	newBias, err := biasCopy.Apply(func(x float64) float64 {
		res := -o.LearningRate * x
		return res
	})
	handleError(err)
	updateBias, err := t.Add(bias, newBias)
	handleError(err)

	return UpdatedWeight, updateBias
}

func (nn *NeuralNetwork) UpdateWeight() {

	for _, hl := range nn.HiddenLayer {
		updatedWeight, updatedBias := nn.Gd.UpdateParameter(hl.Layer.Weight, hl.Layer.DWeight, hl.Layer.Bias, hl.Layer.DBias)
		hl.Layer.Weight = updatedWeight
		hl.Layer.Bias = updatedBias
	}
	updatedWeight, updatedBias := nn.Gd.UpdateParameter(nn.OutPutLayer.Layer.Weight, nn.OutPutLayer.Layer.DWeight, nn.OutPutLayer.Layer.Bias, nn.OutPutLayer.Layer.DBias)
	nn.OutPutLayer.Layer.Weight = updatedWeight
	nn.OutPutLayer.Layer.Bias = updatedBias
}
