package nn

import (
	"strconv"
)

// Layer structure represent one layer in the neural network
// it can be a hidden layer or a input layer
type layer struct {
	nNeuron int
	// activationFunction func([]float32) ([]float32, error)
	hiddenLayer bool
	outputLayer bool
	// weightInitializer  func([]float32) ([]float32, error)
}

// NeuralNetwork structure represent the neural network with his layer,
// and hyperparameter
type NeuralNetwork struct {
	batch        int
	layer        []layer
	learningRate float64
	weight       map[string][][]float64
	bias         map[string][]float64
	data         [][]float64
	// loss         func([]float32, []float32) float32
}

func initWeight(sizeData int, layers []int) (map[string][][]float64, map[string][]float64) {
	nLayers := len(layers)
	weight := make(map[string][][]float64, nLayers)
	bias := make(map[string][]float64, nLayers)
	for idx, value := range layers {
		bias["B"+strconv.Itoa(idx)] = make([]float64, value)
		if idx == 0 {
			weight["W"+strconv.Itoa(idx)] = make([][]float64, value)
			for idxTmp := range value {

				weight["W"+strconv.Itoa(idx)][idxTmp] = make([]float64, sizeData)
			}
			continue
		}
		weight["W"+strconv.Itoa(idx)] = make([][]float64, value)

		for idxTmp := range value {
			weight["W"+strconv.Itoa(idx)][idxTmp] = make([]float64, layers[idx-1])
		}
	}
	return weight, bias
}

// CreateNetwork Permit to create network and initialize the weight for n layer and n neuron in each layer
func (n *NeuralNetwork) CreateNetwork(batch int, layers []int, learningRate float64, data [][]float64) (*NeuralNetwork, error) {
	weight, bias := initWeight(len(data), layers)
	layerSize := len(layers)

	layerTmp := make([]layer, layerSize+1)

	for idx, value := range layers {

		layerTmp[idx] = layer{
			nNeuron:     value,
			hiddenLayer: true,
			outputLayer: false,
		}
	}
	layerTmp[layerSize] = layer{
		nNeuron:     2,
		hiddenLayer: false,
		outputLayer: true,
	}
	return &NeuralNetwork{
		weight: weight,
		layer: layerTmp,
		batch: batch,
		learningRate: learningRate,
		data: data,
		bias: bias,
	},nil

}
