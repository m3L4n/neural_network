package ml

import (
	"fmt"
	"math/rand"

	t "gorgonia.org/tensor"
)

type HiddenLayerStruct struct {
	Layer      *LayerDense
	Activation *ActivationReLu
}

//	type HiddenLayerStruct struct {
//		Layer      *LayerDense
//		Activation *ActivationSigmoid
//	}
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
		layer := NewLayerDense(nNeuronBefore, nNeuron, 0, 0, 5e-4, 5e-4)
		nNeuronBefore = nNeuron
		allLayer[idx].Layer = layer
		allLayer[idx].Activation = NewActivation()
		idx++
	}
	outPutLayer := OutputLayerStruct{
		Layer:      NewLayerDense(nNeuronBefore, 2, 0, 0, 0, 0),
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
	//  ))
	var newX = t.New(t.WithShape(batch_size, shape[1]), t.WithBacking(valueTensor))
	var newy = t.New(t.WithShape(batch_size, 1), t.WithBacking(yTensor))
	return newX, newy
}
func (nn *NeuralNetwork) Fit(X, y, xTest, yTest t.Tensor, batch int) {

	lossTest := make([]float64, nn.epoch)
	accTest := make([]float64, nn.epoch)
	for i := 0; i < nn.epoch; i++ {

		// take batch of X
		x, yNew := takeRandomBatch(X, y, batch)
		input := x

		for _, hLayer := range nn.HiddenLayer {
			input = hLayer.ForwardLayer(input)
		}
		pred := nn.OutPutLayer.ForwardLayer(input)
		backWardOutput := nn.OutPutLayer.BackwardLayer(yNew)
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
		acc := Accuracy(pred, yNew)
		loss, _ := BinaryCrossEntropy(pred, yNew)
		accTest[i] = Accuracy(predTest, yTest)
		lossTest[i] = lossTestVal
		nn.Loss[i] = loss
		nn.Accuracy[i] = acc
		fmt.Printf(" Epoch %v \t Accuracy : %v \t loss : %v \tloss test :  %v acc test %v \n ", i, acc, loss, lossTestVal, accTest[i])
		// if (lossTest[i] <= 0.08){
		// 	fmt.Print("loss of 0.08 get", lossTest[i])
		// }
	}
	PlotData("loss.png", nn.Loss, lossTest)
	PlotData("Accuracy.png", nn.Accuracy, accTest)

}

func (nn *NeuralNetwork) Predict(x, y t.Tensor) {
	input := x
	for _, hLayer := range nn.HiddenLayer {
		input = hLayer.ForwardLayer(input)
	}
	pred := nn.OutPutLayer.ForwardLayer(input)
	loss, err := BinaryCrossEntropy(pred, y)
	handleErrorMsg("Error in binary cross entropy", err)
	acc := Accuracy(pred, y)
	fmt.Printf("Accuracy for the prediction %v \t loss : %v", acc, loss)
}
