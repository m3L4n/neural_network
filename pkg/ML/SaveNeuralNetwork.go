package ml

import (
	"encoding/gob"
	"os"

	t "gorgonia.org/tensor"
)

type LayerConfig struct {
	Type       string
	InputSize  int
	OutputSize int
	// Activation string
	L1Reg     float64
	L2Reg     float64
	Weights   []float64
	Shape     []int
	Bias      []float64
	Shapebias []int
}

func SaveLayerConfigsToBinary(configs []LayerConfig, filepath string) error {
	file, err := os.Create(filepath)
	if err != nil {
		return err
	}
	defer file.Close()

	encoder := gob.NewEncoder(file)
	return encoder.Encode(configs)
}

func LoadLayerConfigsFromBinary(filepath string) ([]LayerConfig, error) {
	file, err := os.Open(filepath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var configs []LayerConfig
	decoder := gob.NewDecoder(file)
	err = decoder.Decode(&configs)
	return configs, err
}

func CreateLayerConfig(name string, inputSize, outputSize int, weightTensor, biasTensor t.Tensor) LayerConfig {
	data := weightTensor.Data().([]float64)
	shape := weightTensor.Shape()
	biases := biasTensor.Data().([]float64)
	biasShape := biasTensor.Shape()
	return LayerConfig{
		Type:       name,
		InputSize:  inputSize,
		OutputSize: outputSize,
		Weights:    data,
		Bias:       biases,
		Shape:      shape,
		Shapebias:  biasShape,
	}
}

func SaveNeuralNetwork(nn NeuralNetwork) {
	var layer = make([]LayerConfig, len(nn.HiddenLayer)+1)

	for i, hl := range nn.HiddenLayer {
		shape := hl.Layer.Weight.Shape()
		layer[i] = CreateLayerConfig("hidden", shape[0], shape[1], hl.Layer.Weight, hl.Layer.Bias)
	}
	shapeOutput := nn.OutPutLayer.Layer.Weight.Shape()
	layer[len(nn.HiddenLayer)] = CreateLayerConfig("output", shapeOutput[0], shapeOutput[1], nn.OutPutLayer.Layer.Weight, nn.OutPutLayer.Layer.Bias)
	filepath := "./model/model.bin"
	err := SaveLayerConfigsToBinary(layer, filepath)
	handleErrorMsg("Error in save weight and bias", err)

}

func LoadNeuralNetwork(path string) NeuralNetwork {
	loadedLayers, err := LoadLayerConfigsFromBinary("./model/model.bin")
	handleErrorMsg("Error in loading", err)
	allLayer := make([]HiddenLayerStruct, len(loadedLayers)-1)
	outPutLayer := OutputLayerStruct{}
	for idx, layer := range loadedLayers {
		if layer.Type == "hidden" {

			newLayer := NewLayerDense(layer.InputSize, layer.OutputSize, 0, 0, 0, 0)
			newLayer.Weight = t.New(t.WithShape(layer.Shape...), t.WithBacking(layer.Weights))
			newLayer.Bias = t.New(t.WithShape(layer.Shapebias...), t.WithBacking(layer.Bias))
			allLayer[idx] = HiddenLayerStruct{Layer: newLayer, Activation: NewActivation()}
		} else if layer.Type == "output" {
			outPutLayer.Activation = NewActivationSoftmax()
			newLayerOuput := NewLayerDense(layer.InputSize, layer.OutputSize, 0, 0, 0, 0)
			newLayerOuput.Weight = t.New(t.WithShape(layer.Shape...), t.WithBacking(layer.Weights))
			newLayerOuput.Bias = t.New(t.WithShape(layer.Shapebias...), t.WithBacking(layer.Bias))
			outPutLayer.Layer = newLayerOuput
		}
	}
	return NeuralNetwork{HiddenLayer: allLayer, OutPutLayer: outPutLayer}
}
