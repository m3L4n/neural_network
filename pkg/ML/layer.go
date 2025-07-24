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

// NewLayerDense creates a new dense layer with the specified number of inputs and neurons.
// It initializes the weights and biases with random values and sets the regularization parameters.
func NewLayerDense(n_input, n_neuron int, lweight_regL1, lwbias_regL1, lweight_regL2, lwbias_regL2 float64) *LayerDense {
	weightCopy := t.New(t.WithShape(n_input, n_neuron), t.WithBacking(t.Random(t.Float64, n_input*n_neuron)))
	stdDev := math.Sqrt(2.0 / float64(n_input+n_neuron))
	weight, err := weightCopy.Apply(func(x float64) float64 {
		return x*2*stdDev - stdDev
	})
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

	// Compute the gradient of the weights: dW = Xᵀ · dZ ((dvalues) gradient of the loss w.r.t. the output of this layer )
	dweight, err := t.Dot(inputTranspose, dvalues)
	handleError(err)

	// Compute the gradient of the biases: db = sum of dZ over the batch (axis 0)
	sum, err := t.Sum(dvalues, 0)
	handleError(err)
	sum.Reshape(1, sum.Shape()[0])
	tmpWeight := l.Weight.Clone().(t.Tensor)
	tmpWeight.T()

	// If L2 regularization is enabled on weights, add 2 * λ * W to the gradient
	if l.Weight_regL2 > 0 {
		tmpWeightL2, err := t.Mul((2 * l.Weight_regL2), l.Weight)

		handleErrorMsg("Error in reg l2 mul scalar", err)
		regDweight, err := t.Add(dweight, tmpWeightL2)
		handleErrorMsg("Error in reg l2 in add tensor", err)
		dweight = regDweight
	}

	// If L2 regularization is enabled on biases, add 2 * λ * b to the gradient
	if l.Bias_regL2 > 0 {
		tmpBias, err := t.Mul((2 * l.Bias_regL2), l.Bias)

		handleErrorMsg("Error in reg l2 mul scalar", err)
		regBias, err := t.Add(sum, tmpBias)
		handleErrorMsg("Error in reg l2 in add tensor", err)
		sum = regBias
	}

	// Store gradients for optimizer
	l.DWeight = dweight
	l.DBias = sum

	// Compute gradient w.r.t. input to pass backward: dInput = dZ · Wᵀ
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
