package gone

import (
	"github.com/fr3fou/gone/matrix"
)

type Layer struct {
	Nodes     int
	Activator Activation
	Bias      float64
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

	for i := 0; i < l-1; i++ {
		prev := layers[i]
		current := layers[i+1]
		weights := matrix.New(
			current.Nodes, // the rows are the inputs of the current one
			prev.Nodes,    // the cols are the outputs of the previous layer
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

	// Set fallbacks
	for i := range layers {
		if layers[i].Activator.F == nil {
			layers[i].Activator.F = Id.F
		}

		if layers[i].Activator.FPrime == nil {
			layers[i].Activator.FPrime = Id.FPrime
		}

		if layers[i].Bias == 0.0 {
			layers[i].Bias = 1
		}
	}

	return n
}

// Predict is the feedforward process
func (n *NeuralNetwork) Predict(data []float64) matrix.Matrix {
	if len(data) != n.Layers[0].Nodes {
		panic("gone: not enough data in input layer")
	}

	output := matrix.NewFromArray(data)

	for i := 0; i < len(n.Weights); i++ {
		output = n.Weights[i].
			Multiply(output).                         // weighted sum of the previous layer)
			Add(n.Layers[i+1].Bias).                  // bias
			Map(func(val float64, x, y int) float64 { // activation
				return n.Layers[i+1].Activator.F(val)
			})
	}

	return output
}
