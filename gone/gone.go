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
		Weights:      make([]matrix.Matrix, l-1),
		Errors:       make([]matrix.Matrix, l-1),
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
	}

	for i := range layers {
		if layers[i].ActivationFunction.F == nil {
			layers[i].ActivationFunction.F = Id.F
		}

		if layers[i].ActivationFunction.FPrime == nil {
			layers[i].ActivationFunction.FPrime = Id.FPrime
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
