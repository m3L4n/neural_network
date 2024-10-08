package nn

import "errors"

type layer struct {
	nNeuron            uint
	activationFunction func([]float32) ([]float32, error)
	hiddenLayer        bool
	outputLayer        bool
	weightInitializer  func([]float32) ([]float32, error)
}

type neuralNetwork struct {
	batch        uint16
	layer        []layer
	learningRate float32
	weight       []float32
	loss         func([]float32, []float32) float32
}

func (n *neuralNetwork) createNetwork(batch uint16, layers []layer, learningRate float32, loss func([]float32, []float32) float32) (*neuralNetwork, error) {
	if len(layers) <= 0 {
		return nil, errors.New("layers need to be > 0")
	}
	return &neuralNetwork{
		batch:        batch,
		learningRate: learningRate,
		layer:        layers,
		loss:         loss,
	}, nil

}
