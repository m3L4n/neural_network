package nn

import "errors"

type layer struct {
	n_neuron            uint
	activation_function func([]float32) ([]float32, error)
	hidden_layer        bool
	output_layer        bool
	weight_initializer  func([]float32) ([]float32, error)
}

type neural_network struct {
	batch         uint16
	layer         []layer
	learning_rate float32
	weight        []float32
	loss func([]float32, []float32) float32
}

func (n *neural_network) createNetwork(batch uint16, layers []layer, learning_rate float32 ,loss func([]float32, []float32) float32) (*neural_network, error){
	if (len(layers) <= 0){
		return nil,errors.New("layers need to be > 0")
	}
return &neural_network{
	batch: batch,
	learning_rate: learning_rate,
	layer: layers,
	loss:loss ,
}, nil

}
