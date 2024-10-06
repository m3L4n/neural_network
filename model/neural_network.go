package model

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
}

// func () createNetwork() n
