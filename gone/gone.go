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
}

func New(lr float64, layers ...Layer) *NeuralNetwork {
	l := len(layers)
	n := &NeuralNetwork{
		Weights:      make([]matrix.Matrix, l),
		Errors:       make([]matrix.Matrix, l),
		LearningRate: lr,
	}

	for i := 1; i <= l; i++ {
		weights := matrix.New(
			layers[i-1].Nodes, // the rows are the outputs of the previous layer
			layers[i].Nodes,   // the cols are the inputs of the current one
			nil,
		)
		weights.Randomize(-1, 2) // Initialize the weights randomly
		n.Weights[i] = weights

		errors := matrix.New(
			layers[i-1].Nodes, // the rows are the outputs of the previous layer
			1,
			nil,
		)
		n.Errors[i] = errors
	}

	return n
}
