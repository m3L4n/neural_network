package ml

import (
	"errors"
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

func NewLayerDense(n_input, n_neuron int) LayerDense {
	weightCopy := t.New(t.WithShape(n_input, n_neuron), t.WithBacking(t.Random(t.Float64, n_input*n_neuron)))
	stdDev := math.Sqrt(2.0 / float64(n_input))
	weight, err := weightCopy.MulScalar(stdDev, false)
	handleError(err)
	bias := t.New(t.WithShape(1, n_neuron), t.Of(t.Float64))
	return LayerDense{Weight: weight, Bias: bias, input: t.New(t.Of(t.Float64))}
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

func BinaryCrossEntropy(yPred, y t.Tensor) (float64, error) {

	newYPred := Argmax(yPred)
	shapeY := y.Shape()
	shapeYPred := newYPred.Shape()
	if shapeY[0] != shapeYPred[0] {
		return 0, errors.New("shape of prediction and labels have to be the same")
	}
	epsilon := 1e-10
	// newYPredClipped := newYPred.Clone().(t.Tensor)
	// for i := 0; i < newYPred.Shape()[0]; i++{
	// 	value , err := newYPred.At(i, 0)
	// 	handleError(err)
	// 	if (value.(float64) < epsilon){

	// 	newYPredClipped.SetAt(epsilon, i, 0)
	// 	}else if (value.(float64) > 1- epsilon ){
	// 		newYPredClipped.SetAt(1.0 - epsilon, i, 0)
	// 	}else{
	// 	newYPredClipped.SetAt(value.(float64), i, 0)
	// 	}
	// }
	var bce = 0.0
	for i := 0; i < shapeY[0]; i++ {
		valueLabel, err := y.At(i, 0)
		valueLabelF := valueLabel.(float64)
		handleError(err)
		valuePrediction, err := newYPred.At(i, 0)
		valuePredictionF := valuePrediction.(float64)
		handleError(err)
		bce += valueLabelF*math.Log(math.Max(valuePredictionF, epsilon)) + (1-valueLabelF)*math.Log(math.Max(1-valuePredictionF, epsilon))
	}

	return -bce / float64(shapeY[0]), nil
}

func Argmax(yPred t.Tensor) t.Tensor {
	shapeYPred := yPred.Shape()
	fmt.Println("ARGMAX", shapeYPred)
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
	for idx := 0 ; idx <  yTrue.Shape()[0]; idx++{
		fmt.Println(newYPred.At(idx,0))
	}
	fmt.Println(newYPred.Shape()[0], yTrue.Shape()[0])
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
