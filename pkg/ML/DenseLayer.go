package ml

import (
	"fmt"
	"log"
	"math"

	t "gorgonia.org/tensor"
)

type LayerDense struct {
	Weight  t.Tensor
	Bias    t.Tensor
	Output  t.Tensor
	input   t.Tensor
	DWeight t.Tensor
	DBias   t.Tensor
	DInput  t.Tensor
}

func addBias(inputs t.Tensor, bias t.Tensor) t.Tensor {

	shape := inputs.Shape()
	row := shape[0]
	column := shape[1]
	var biasT *t.Dense = bias.Clone().(*t.Dense)
	biasT.T()
	newResult := t.New(t.WithShape(row, column), t.Of(t.Float64))
	for j := 0; j < row; j++ {
		for i := 0; i < column; i++ {
			value, _ := inputs.At(j, i)
			valueBias, _ := biasT.At(i, 0)
			res := value.(float64) + valueBias.(float64)
			newResult.SetAt(res, j, i)
		}
	}
	return newResult

}

func NewLayerDense(n_input, n_neuron int) *LayerDense {
	weightCopy := t.New(t.WithShape(n_input, n_neuron), t.WithBacking(t.Random(t.Float64, n_input*n_neuron)))
	stdDev := math.Sqrt(2.0 / float64(n_input))
	weight, err := weightCopy.Apply(func(x float64) float64 {
		// return x*0.2 - 0.1
		return x*2*stdDev - stdDev // Échelle uniforme entre [-stdDev, stdDev]
	})
	// stdDev := math.Sqrt(6.0 / float64(n_neuron+n_input))
	// weight, err := weightCopy.MulScalar((stdDev * 2), false)
	// weightSub, _ := weight.SubScalar(stdDev, false)
	handleError(err)
	bias := t.New(t.WithShape(1, n_neuron), t.Of(t.Float64))
	return &LayerDense{Weight: weight, Bias: bias, input: t.New(t.Of(t.Float64))}
}

func (l *LayerDense) Foward(inputs t.Tensor) {
	l.input = inputs
	dP, err := t.Dot(inputs, l.Weight)
	handleError(err)
	if len(dP.Shape()) == 1 {
		dP.Reshape(dP.Shape()[0], 1)
	}
	layerOutput := addBias(dP, l.Bias)
	l.Output = layerOutput
}

func (l *LayerDense) Backward(dvalues t.Tensor) {

	var inputTranspose t.Tensor = l.input.Clone().(t.Tensor)
	inputTranspose.T()
	dweight, err := t.Dot(inputTranspose, dvalues)
	l.DWeight = dweight
	handleError(err)
	sum, err := t.Sum(dvalues, 0)
	handleError(err)
	sum.Reshape(1, sum.Shape()[0])
	l.DBias = sum
	tmpWeight := l.Weight.Clone().(t.Tensor)
	tmpWeight.T()

	l.DInput, err = t.Dot(dvalues, tmpWeight)
	handleError(err)

}

func handleError(err error) {
	if err != nil {
		log.Fatal(err)
		fmt.Println(err)
	}
}

func Argmax(yPred t.Tensor) t.Tensor {
	shapeYPred := yPred.Shape()
	newYPred := t.New(t.WithShape(yPred.Shape()[0], 1), t.Of(t.Float64))
	for i := 0; i < shapeYPred[0]; i++ {
		indexMax := 0
		valueMax, err := yPred.At(i, indexMax)
		handleError(err)
		for j := 0; j < shapeYPred[1]; j++ {
			valueTmp, err := yPred.At(i, j)
			handleError(err)
			if (valueMax.(float64)) < valueTmp.(float64) {
				valueMax = valueTmp
				indexMax = j
			}
		}
		newYPred.SetAt(float64(indexMax), i, 0)
	}
	return newYPred
}

func Accuracy(yPred, yTrue t.Tensor) float64 {

	newYPred := Argmax(yPred)
	for idx := 0; idx < yTrue.Shape()[0]; idx++ {
	}
	// fmt.Println(newYPred.Shape()[0], yTrue.Shape()[0])
	predictionSum := 0
	for idx := 0; idx < yTrue.Shape()[0]; idx++ {
		classPred, err := newYPred.At(idx, 0)
		handleError(err)
		classTrue, err := yTrue.At(idx, 0)
		handleError(err)
		if classPred.(float64) == classTrue.(float64) {
			predictionSum += 1
		} else {
			predictionSum += 0
		}
	}
	var meanAccuracy float64 = float64(predictionSum) / float64(yTrue.Shape()[0])
	return meanAccuracy
}

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
