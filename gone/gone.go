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
	Activations  []matrix.Matrix
	LearningRate float64
	Layers       []Layer
	BatchSize    int
	Task         Task
	// Loss         Loss
}

func New(learningRate float64, batchSize int, task Task,
	// loss Loss,
	layers ...Layer) *NeuralNetwork {
	l := len(layers)
	if l < 3 { // minimum amount of layers
		panic("gone: need more layers for a neural network")
	}
	n := &NeuralNetwork{
		Weights:     make([]matrix.Matrix, l-1),
		Activations: make([]matrix.Matrix, l-1),
		Task:        task,
		// Loss:         loss,
		Layers:       layers,
		BatchSize:    batchSize,
		LearningRate: learningRate,
	}

	for i := 0; i < l-1; i++ {
		current := layers[i]
		next := layers[i+1]
		weights := matrix.New(
			next.Nodes,    // the rows are the inputs of the current one
			current.Nodes, // the cols are the outputs of the previous layer
			nil,
		)
		weights.Randomize(-1, 2) // Initialize the weights randomly
		n.Weights[i] = weights

		n.Activations[i] = matrix.New(current.Nodes, 1, nil)
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
func (n *NeuralNetwork) predict(mat matrix.Matrix) matrix.Matrix {
	for i := 0; i < len(n.Weights); i++ {
		mat = n.Weights[i].
			DotProduct(mat).                          // weighted sum of the previous layer)
			Add(n.Layers[i+1].Bias).                  // bias
			Map(func(val float64, x, y int) float64 { // activation
				return n.Layers[i+1].Activator.F(val)
			})
		n.Activations[i] = mat
	}

	return mat
}

// DataSet represents a slice of all the entires in a data set
type DataSet []DataSample

// DataSample represents a single train data set
type DataSample struct {
	Inputs  []float64
	Targets []float64
}

// Shuffle shuffles the data in a random order
func (t DataSet) Shuffle() {
	rand.Shuffle(len(t), func(i, j int) {
		t[i], t[j] = t[j], t[i]
	})
}

// Batch chunks the slice
func (t DataSet) Batch(current int, batchSize int) DataSet {
	length := len(t)
	end := current + batchSize

	if current >= length || current == -1 || end > length {
		return DataSet{}
	}

	return t[current:end] // return the current chunk
}

// Train trains the neural network using backpropagation
func (n *NeuralNetwork) Train(dataSet DataSet, epochs int) {
	layerAmount := len(n.Layers)
	weightAmount := len(n.Weights)
	activationAmount := weightAmount
	inputNodes := n.Layers[0].Nodes
	outputNodes := n.Layers[layerAmount-1].Nodes

	// Check if the user has provided enough inputs
	for _, dataCase := range dataSet {
		if len(dataCase.Inputs) != inputNodes {
			panic("gone: not enough data in input layer")
		}

		if len(dataCase.Targets) != outputNodes {
			panic("gone: not enough labels in output layer")
		}
	}

	for epoch := 0; epoch < epochs; epoch++ {

		for i, data := range dataSet {
			currentInputs := matrix.NewFromArray(data.Inputs)
		}
	}
}

func (n *NeuralNetwork) backpropagate(ds DataSample) {

}
