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
	BatchSize    int
	Task         Task
}

func New(alpha float64, b int, task Task, layers ...Layer) *NeuralNetwork {
	l := len(layers)
	if l < 3 { // minimum amount of layers
		panic("gone: need more layers for a neural network")
	}
	n := &NeuralNetwork{
		Weights:      make([]matrix.Matrix, l-1),
		Errors:       make([]matrix.Matrix, l-1),
		Task:         task,
		Layers:       layers,
		BatchSize:    b,
		LearningRate: alpha,
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
func (n *NeuralNetwork) predict(mat matrix.Matrix) matrix.Matrix {
	for i := 0; i < len(n.Weights); i++ {
		mat = n.Weights[i].
			DotProduct(mat).                          // weighted sum of the previous layer)
			Add(n.Layers[i+1].Bias).                  // bias
			Map(func(val float64, x, y int) float64 { // activation
				return n.Layers[i+1].Activator.F(val)
			})
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
	// Check if the user has provided enough inputs
	inputNodes := n.Layers[0].Nodes
	outputNodes := n.Layers[len(n.Layers)-1].Nodes

	for _, dataCase := range dataSet {
		if len(dataCase.Inputs) != inputNodes {
			panic("gone: not enough data in input layer")
		}

		if len(dataCase.Targets) != outputNodes {
			panic("gone: not enough labels in output layer")
		}
	}

	for epoch := 0; epoch < epochs; epoch++ {
		for i := 0; i < n.BatchSize; i += n.BatchSize {
			// Batch for Batch Gradient Descent
			batch := dataSet.Batch(i, n.BatchSize)

			// Shuffle the data
			batch.Shuffle()

			// Output matrix
			outputs := matrix.New(n.BatchSize, outputNodes, nil)

			// Target matrix
			targets := matrix.New(n.BatchSize, outputNodes, nil)

			// Go through the batch and store the predictions
			for _, data := range batch {
				currentInputs := matrix.NewFromArray(data.Inputs)
				outputs.Data[i] = n.predict(currentInputs).Flatten()
			}

			// Nx1 Matrix (N being the number of output nodes)
			// Compute the errors
			errors := mse(outputs, targets)
			// .Map(func(val float64, x, y int) float64 {
			// 	// Calculate the gradients
			// 	return n.Layers[len(n.Layers)-1].Activator.FPrime(val)
			// })

			// Backpropagate
			for i := len(n.Weights); i > 0; i-- {
				// The previous layer's weights but transposed
				transposed := n.Weights[i-1].
					Transpose().
					Map(func(val float64, x, y int) float64 {
						// Calculate the gradients
						return n.Layers[len(n.Layers)-1].Activator.FPrime(val)
					})

				// Update the errors with the current layer's errors
				errors = transposed.HadamardProduct(errors)

				// Calculate deltas
				deltas := errors.DotProduct(transposed)

				// Adjust the weights
				n.Weights[i-1] = n.Weights[i-1].AddMatrix(deltas)
			}
		}
	}
}

func mse(outputs, targets matrix.Matrix) matrix.Matrix {
	// Calculate the error
	// Error_{i,j} = Target_{i,j} - Outputs_{i,j}
	errs := targets.SubtractMatrix(outputs)

	// Raise them to the 2nd power
	squared := errs.DotProduct(errs)

	// Make a new one dimensional vector of all the mean errors
	return matrix.Map(matrix.New(squared.Columns, 1, nil), func(val float64, x, y int) float64 {
		sum := 0.0
		for i := 0; i < squared.Columns; i++ {
			sum += squared.Data[x][i]
		}

		return sum / float64(squared.Columns)
	})
}
