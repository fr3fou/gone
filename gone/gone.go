package gone

import (
	"math/rand"

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
func (n *NeuralNetwork) Predict(data []float64) []float64 {
	if len(data) != n.Layers[0].Nodes {
		panic("gone: not enough data in input layer")
	}

	return n.predict(matrix.NewFromArray(data)).Flatten()
}

// predict is a helper function that uses matricies instead of slices
func (n *NeuralNetwork) predict(input matrix.Matrix) matrix.Matrix {
	output := input

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

// DataSet represents a slice of all the entires in a data set
type DataSet []DataCase

// DataCase represents a single train data set
type DataCase struct {
	Inputs  []float64
	Targets []float64
}

// Shuffle shuffles the data in a random order
func (t DataSet) Shuffle() {
	rand.Shuffle(len(t), func(i, j int) {
		t[i], t[j] = t[j], t[i]
	})
}

// Train trains the neural network using backpropagation
func (n *NeuralNetwork) Train(dataSet DataSet, epochs int) {
	// Check if the user has provided enough inputs
	inputLayer := n.Layers[0]
	outputLayer := n.Layers[len(n.Layers)-1]

	for _, dataCase := range dataSet {
		if len(dataCase.Inputs) != inputLayer.Nodes {
			panic("gone: not enough data in input layer")
		}

		if len(dataCase.Targets) != outputLayer.Nodes {
			panic("gone: not enough labels in output layer")
		}
	}

	for epoch := 0; epoch < epochs; epoch++ {
		// Shuffle the data
		dataSet.Shuffle()

		for _, data := range dataSet {
			inputs := matrix.NewFromArray(data.Inputs)
			targets := matrix.NewFromArray(data.Targets)
			outputs := n.predict(inputs)

			// Calculate the error
			// E =
			errors := targets.SubtractMatrix(outputs)
		}
	}
}
