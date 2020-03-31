package gone

import (
	"log"
	"math/rand"

	"github.com/fr3fou/gone/matrix"
)

// Layer represents a layer in a neural network
type Layer struct {
	Nodes     int
	Activator Activation
}

// NeuralNetwork represents a neural network
type NeuralNetwork struct {
	Weights      []matrix.Matrix
	Biases       []matrix.Matrix
	Activations  []matrix.Matrix
	LearningRate float64
	Layers       []Layer
	DebugMode    bool
	BatchSize    int
	// Loss         Loss
}

// New creates a neural network
func New(learningRate float64 /* loss Loss */, layers ...Layer) *NeuralNetwork {
	l := len(layers)
	if l < 3 { // minimum amount of layers
		panic("gone: need more layers for a neural network")
	}
	n := &NeuralNetwork{
		Weights:     make([]matrix.Matrix, l-1),
		Biases:      make([]matrix.Matrix, l-1),
		Activations: make([]matrix.Matrix, l),
		// Loss:         loss,
		Layers: layers,
		// BatchSize:    batchSize,
		LearningRate: learningRate,
	}

	// Initialize the weights and biases
	for i := 0; i < l-1; i++ {
		current := layers[i]
		next := layers[i+1]
		weights := matrix.New(
			next.Nodes,    // the rows are the inputs of the next one
			current.Nodes, // the cols are the outputs of the current layer
			nil,
		)
		weights.Randomize(-1, 2) // Initialize the weights randomly
		n.Weights[i] = weights

		biases := matrix.New(
			next.Nodes, // the rows are the inputs of the next one
			1,
			nil,
		)
		biases.Randomize(-1, 2) // Initialize the biases randomly
		n.Biases[i] = biases
	}

	// Initialize the activations
	for i := 0; i < l; i++ {
		current := layers[i]
		n.Activations[i] = matrix.New(current.Nodes, 1, nil)
	}

	// Set fallbacks
	for i := range layers {
		if layers[i].Activator.F == nil || layers[i].Activator.FPrime == nil {
			layers[i].Activator = Identity()
		}
	}

	return n
}

func (n *NeuralNetwork) ToggleDebug(b bool) {
	n.DebugMode = b
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
	n.Activations[0] = mat // add the original input
	for i := 0; i < len(n.Weights); i++ {
		mat = n.Weights[i].
			DotProduct(mat).                          // weighted sum of the previous layer)
			AddMatrix(n.Biases[i]).                   // bias
			Map(func(val float64, x, y int) float64 { // activation
				return n.Layers[i+1].Activator.F(val)
			})
		n.Activations[i+1] = mat.Copy()
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
func (n *NeuralNetwork) Train(optimizer Optimizer, dataSet DataSet, epochs int) {
	inputNodes := n.Layers[0].Nodes
	outputNodes := n.Layers[len(n.Layers)-1].Nodes

	// Check if the user has provided enough inputs
	for _, dataCase := range dataSet {
		if len(dataCase.Inputs) != inputNodes {
			panic("gone: not enough data in input layer")
		}

		if len(dataCase.Targets) != outputNodes {
			panic("gone: not enough labels in output layer")
		}
	}

	for i := 1; i <= epochs; i++ {
		if n.DebugMode {
			log.Printf("Beginning epoch %d/%d", i, epochs)
		}
		optimizer(n, dataSet)
		if n.DebugMode {
			err := 0.0
			for _, ds := range dataSet {
				input := matrix.NewFromArray(ds.Inputs)
				output := n.predict(input)

				currentError := matrix.NewFromArray(ds.Targets).SubtractMatrix(output)
				currentError = output.HadamardProduct(output) // Squared
				// for some reason squaring it makes it worse? it converges to 0.5 instead of 0

				err += currentError.Fold(
					func(accumulator, val float64, x, y int) float64 {
						return val + accumulator
					},
					0,
				)
				err /= float64(outputNodes)
				err /= float64(len(dataSet))
			}

			log.Printf("Finished epoch %d/%d with error: %f", i, epochs, err)
		}
	}
}
