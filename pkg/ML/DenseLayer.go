package ml

import (
	"fmt"
	"log"
	"math"

	t "gorgonia.org/tensor"
)

type LayerDense struct {
	Weight       t.Tensor
	Bias         t.Tensor
	Output       t.Tensor
	input        t.Tensor
	DWeight      t.Tensor
	DBias        t.Tensor
	DInput       t.Tensor
	Weight_regL2 float64
	Weight_regL1 float64
	Bias_regL1   float64
	Bias_regL2   float64
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

func NewLayerDense(n_input, n_neuron int, lweight_regL1, lwbias_regL1, lweight_regL2, lwbias_regL2 float64) *LayerDense {
	weightCopy := t.New(t.WithShape(n_input, n_neuron), t.WithBacking(t.Random(t.Float64, n_input*n_neuron)))
	// stdDev := math.Sqrt( float64(n_input))
	stdDev := math.Sqrt(2.0 / float64(n_input + n_neuron))
	weight, err := weightCopy.Apply(func(x float64) float64 {
		return x*2*stdDev - stdDev
		// return x / stdDev
		// return x / stdDev
	})
	// stdDev := math.Sqrt(6.0 / float64(n_neuron+n_input))
	// weight, err := weightCopy.MulScalar((stdDev * 2), false)
	// weightSub, _ := weight.SubScalar(stdDev, false)
	handleError(err)
	bias := t.New(t.WithShape(1, n_neuron), t.Of(t.Float64))
	return &LayerDense{Weight: weight, Bias: bias, input: t.New(t.Of(t.Float64)), Weight_regL1: lweight_regL1, Bias_regL1: lwbias_regL1, Weight_regL2: lweight_regL2, Bias_regL2: lwbias_regL2}
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
	handleError(err)
	sum, err := t.Sum(dvalues, 0)
	handleError(err)
	sum.Reshape(1, sum.Shape()[0])
	tmpWeight := l.Weight.Clone().(t.Tensor)
	tmpWeight.T()
	if l.Weight_regL2 > 0 {
		tmpWeightL2, err := t.Mul((2 * l.Weight_regL2), l.Weight)

		handleErrorMsg("Error in reg l2 mul scalar", err)
		regDweight, err := t.Add(dweight, tmpWeightL2)
		handleErrorMsg("Error in reg l2 in add tensor", err)
		dweight = regDweight
	}

	if l.Bias_regL2 > 0 {
		tmpBias, err := t.Mul((2 * l.Bias_regL2), l.Bias)

		handleErrorMsg("Error in reg l2 mul scalar", err)
		regBias, err := t.Add(sum, tmpBias)
		handleErrorMsg("Error in reg l2 in add tensor", err)
		sum = regBias
	}

	l.DWeight = dweight
	l.DBias = sum
	l.DInput, err = t.Dot(dvalues, tmpWeight)
	handleError(err)

}

func handleError(err error) {
	if err != nil {
		log.Fatal(err)
		fmt.Println(err)
	}
}

func handleErrorMsg(msg string, err error) {
	if err != nil {
		log.Fatalf(" error : %v \t%v", msg, err)
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
