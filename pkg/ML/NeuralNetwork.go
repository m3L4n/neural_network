package ml

import (
	"fmt"

	t "gorgonia.org/tensor"
)

type HiddenLayerStruct struct {
	Layer      *LayerDense
	Activation *ActivationReLu
}
type OutputLayerStruct struct {
	Layer      *LayerDense
	Activation *ActivationSoftmax
}
type NeuralNetwork struct {
	HiddenLayer []HiddenLayerStruct
	OutPutLayer OutputLayerStruct
	Gd          *OptimizerGd
	Loss        []float64
	Accuracy    []float64
	epoch       int
}

func NewNeuralNetwork(learningRate float64, X t.Tensor, layer []int, epoch int) NeuralNetwork {
	shape := X.Shape()
	allLayer := make([]HiddenLayerStruct, len(layer))
	var nNeuronBefore int = shape[1]
	// fmt.Println(nNeuronBefore)
	idx := 0
	for _, nNeuron := range layer {
		layer := NewLayerDense(nNeuronBefore, nNeuron)
		nNeuronBefore = nNeuron
		allLayer[idx].Layer = layer
		allLayer[idx].Activation = NewActivation()
		idx++
	}
	outPutLayer := OutputLayerStruct{
		Layer:      NewLayerDense(nNeuronBefore, 2),
		Activation: NewActivationSoftmax(),
	}
	return NeuralNetwork{HiddenLayer: allLayer, OutPutLayer: outPutLayer, Gd: NewOptimizerGd(learningRate), epoch: epoch, Loss: make([]float64, epoch), Accuracy: make([]float64, epoch)}

}

func (hl *HiddenLayerStruct) ForwardLayer(X t.Tensor) t.Tensor {
	hl.Layer.Foward(X)
	hl.Activation.Forward(hl.Layer.Output)
	return hl.Activation.Output
}

func (hl *HiddenLayerStruct) BackwardLayer(X t.Tensor) t.Tensor {
	hl.Activation.Backward(X)
	hl.Layer.Backward(hl.Activation.DInput)

	return hl.Layer.DInput
}

func (oL *OutputLayerStruct) ForwardLayer(X t.Tensor) t.Tensor {
	oL.Layer.Foward(X)
	oL.Activation.Forward(oL.Layer.Output)
	return oL.Activation.Outpout
}
func (oL *OutputLayerStruct) BackwardLayer(y t.Tensor) t.Tensor {
	oL.Activation.Backward(oL.Activation.Outpout, y)
	oL.Layer.Backward(oL.Activation.DInput)
	return oL.Layer.DInput
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
func (nn *NeuralNetwork) Fit(X, y, xTest, yTest t.Tensor) {

	lossTest := make([]float64, nn.epoch)
	accTest := make([]float64, nn.epoch)
	for i := 0; i < nn.epoch; i++ {
		input := X
		for _, hLayer := range nn.HiddenLayer {
			input = hLayer.ForwardLayer(input)
		}
		pred := nn.OutPutLayer.ForwardLayer(input)
		backWardOutput := nn.OutPutLayer.BackwardLayer(y)
		for i := len(nn.HiddenLayer) - 1; i >= 0; i-- {
			if i == -1 {
				break
			}
			backWardOutput = nn.HiddenLayer[i].BackwardLayer(backWardOutput)
		}
		inputTest := xTest
		for _, hLayer := range nn.HiddenLayer {
			inputTest = hLayer.ForwardLayer(inputTest)
		}
		predTest := nn.OutPutLayer.ForwardLayer(inputTest)
		lossTestVal, _ := BinaryCrossEntropy(predTest, yTest)

		nn.UpdateWeight()
		acc := Accuracy(pred, y)
		loss, _ := BinaryCrossEntropy(pred, y)
		lossTest[i] = lossTestVal
		accTest[i] = Accuracy(predTest, yTest)
		nn.Loss[i] = loss
		nn.Accuracy[i] = acc
		fmt.Printf(" Epoch %v \t Accuracy : %v \t loss : %v \tloss test :  %v acc test %v \n ", i, acc, loss, lossTestVal, accTest[i])
	}
	PlotData("loss.png", nn.Loss, lossTest)
	PlotData("Accuracy.png", nn.Accuracy, accTest)

}
