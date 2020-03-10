package gone

import (
	"github.com/fr3fou/gone/matrix"
)

type Layer struct {
	Nodes              int
	ActivationFunction Activation
}

type NeuralNetwork struct {
	Weights      []matrix.Matrix
	Errors       []matrix.Matrix
	LearningRate float64
	Layers       []Layer
}

func New(lr float64, layers ...Layer) *NeuralNetwork {
	l := len(layers)
	if l < 3 { // minimum amount of layers
		panic("gone: need more layers for a neural network")
	}
	n := &NeuralNetwork{
		Weights:      make([]matrix.Matrix, l),
		Errors:       make([]matrix.Matrix, l),
		Layers:       layers,
		LearningRate: lr,
	}

	for i := 1; i < l; i++ {
		prev := layers[i-1]
		current := layers[i]
		weights := matrix.New(
			prev.Nodes,    // the rows are the outputs of the previous layer
			current.Nodes, // the cols are the inputs of the current one
			nil,
		)
		weights.Randomize(-1, 2) // Initialize the weights randomly
		n.Weights[i] = weights

		errors := matrix.New(
			prev.Nodes, // the rows are the outputs of the previous layer
			1,
			nil,
		)
		n.Errors[i] = errors

		// fallback to Id
		if prev.ActivationFunction.F == nil || prev.ActivationFunction.FPrime == nil {
			prev.ActivationFunction = Id
		}
	}

	return n
}
