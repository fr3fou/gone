package gone

import (
	"github.com/fr3fou/gone/matrix"
)

type Layer struct {
	Nodes              int
	ActivationFunction Activation
}

type Task string

const (
	Classification Task = "Classification"
	Regression     Task = "Regression"
)

type NeuralNetwork struct {
	Weights      []matrix.Matrix
	Errors       []matrix.Matrix
	LearningRate float64
	Layers       []Layer
	Task         Task
}

func New(lr float64, task Task, layers ...Layer) *NeuralNetwork {
	l := len(layers)
	if l < 3 { // minimum amount of layers
		panic("gone: need more layers for a neural network")
	}
	n := &NeuralNetwork{
		Weights:      make([]matrix.Matrix, l),
		Errors:       make([]matrix.Matrix, l),
		Task:         task,
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
		if prev.ActivationFunction.F == nil {
			prev.ActivationFunction.F = Id.F
		}

		if prev.ActivationFunction.FPrime == nil {
			prev.ActivationFunction.FPrime = Id.FPrime
		}
	}

	return n
}

// Predict is the feedforward process
func (n *NeuralNetwork) Predict(data []float64) {
	if len(data) != n.Layers[0].Nodes {
		panic("gone: not enough data in input layer")
	}

	input := matrix.NewFromArray(data)
	for i := range n.Layers {

	}
}
