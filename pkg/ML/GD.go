package ml

import (
	"math/rand"

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

func takeRandomBatch(x, y t.Tensor, batch_size int) (t.Tensor, t.Tensor) {
	shape := x.Shape()
	var valueTensor []float64
	var yTensor []float64
	for _ = range batch_size {
		intRand := rand.Intn(shape[0])
		row, err := x.Slice(t.S(intRand))
		rowY, errY := y.Slice(t.S(intRand))
		handleErrorMsg("Error in slice", err)
		handleErrorMsg("Error in slice y ", errY)
		valueTensor = append(valueTensor, row.Data().([]float64)...)
		yTensor = append(yTensor, float64(rowY.Data().(float64)))
	}
	var newX = t.New(t.WithShape(batch_size, shape[1]), t.WithBacking(valueTensor))
	var newy = t.New(t.WithShape(batch_size, 1), t.WithBacking(yTensor))
	return newX, newy
}
