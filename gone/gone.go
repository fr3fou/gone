package gone

import (
	"math/rand"

	"github.com/fr3fou/gone/matrix"
)

// Layer represents a layer in a neural network
type Layer struct {
	Nodes     int
	Activator Activation
}

// Task is the type of task to do
type Task string

const (
	// Classification does a classification
	Classification Task = "Classification"
	// Regression does a regression
	Regression Task = "Regression"
)

// NeuralNetwork represents a neural network
type NeuralNetwork struct {
	InputNodes   int
	HiddenNodes  int
	OutputsNodes int

	InputHiddenWeights  matrix.Matrix
	HiddenOutputWeights matrix.Matrix

	HiddenBias matrix.Matrix
	OutputBias matrix.Matrix

	Activation Activation

	LearningRate float64
	BatchSize    int
	Task         Task
}

// New creates a neural network
func New(learningRate float64, nodes [3]int, activation Activation) *NeuralNetwork {
	n := &NeuralNetwork{
		// BatchSize:    batchSize,
		Activation:   activation,
		LearningRate: learningRate,
	}

	n.InputNodes = nodes[0]
	n.HiddenNodes = nodes[1]
	n.OutputsNodes = nodes[2]

	n.InputHiddenWeights = matrix.New(n.HiddenNodes, n.InputNodes, nil)
	n.HiddenOutputWeights = matrix.New(n.OutputsNodes, n.HiddenNodes, nil)

	n.InputHiddenWeights.Randomize(-1, 2)
	n.InputHiddenWeights.Randomize(-1, 2)

	n.HiddenBias = matrix.New(n.HiddenNodes, 1, nil)
	n.OutputBias = matrix.New(n.OutputsNodes, 1, nil)

	n.HiddenBias.Randomize(-1, 2)
	n.OutputBias.Randomize(-1, 2)

	return n
}

// Predict is the feedforward process
func (n *NeuralNetwork) Predict(data []float64) []float64 {
	if len(data) != n.InputNodes {
		panic("gone: not enough data in input layer")
	}

	return n.predict(matrix.NewFromArray(data)).Flatten()
}

// predict is a helper function that uses matricies instead of slices
func (n *NeuralNetwork) predict(mat matrix.Matrix) matrix.Matrix {
	hidden := n.InputHiddenWeights.
		DotProduct(mat).
		AddMatrix(n.HiddenBias).
		Map(func(val float64, x, y int) float64 {
			return n.Activation.F(val)
		})

	return n.HiddenOutputWeights.
		DotProduct(hidden).
		AddMatrix(n.OutputBias).
		Map(func(val float64, x, y int) float64 {
			return n.Activation.F(val)
		})
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
	inputNodes := n.InputNodes
	outputNodes := n.OutputsNodes

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
		dataSet.Shuffle()
		for _, data := range dataSet {
			// Stochastic Gradient Descent (On-line Training)
			n.backpropagate(data)
		}
	}
}

func (n *NeuralNetwork) backpropagate(ds DataSample) {
	// Prediction
	inputs := matrix.NewFromArray(ds.Inputs)
	hidden := n.InputHiddenWeights.
		DotProduct(inputs).
		AddMatrix(n.HiddenBias).
		Map(func(val float64, x, y int) float64 {
			return n.Activation.F(val)
		})

	outputs := n.HiddenOutputWeights.
		DotProduct(hidden).
		AddMatrix(n.OutputBias).
		Map(func(val float64, x, y int) float64 {
			return n.Activation.F(val)
		})

	// Backpropagation
	targets := matrix.NewFromArray(ds.Targets)

	outputErrors := targets.SubtractMatrix(outputs)
	outputGradients := outputs.
		Map(func(val float64, x, y int) float64 {
			return n.Activation.FPrime(val)
		}).
		HadamardProduct(outputErrors).
		Scale(n.LearningRate)

	hiddenOutputDeltas := outputGradients.
		DotProduct(hidden.Transpose())
	n.HiddenOutputWeights = n.HiddenOutputWeights.
		AddMatrix(hiddenOutputDeltas)
	n.OutputBias = n.OutputBias.
		AddMatrix(outputGradients)

	hiddenErrors := n.HiddenOutputWeights.
		Transpose().
		DotProduct(outputErrors)
	hiddenGradients := hidden.
		Map(func(val float64, x, y int) float64 {
			return n.Activation.FPrime(val)
		}).
		HadamardProduct(hiddenErrors).
		Scale(n.LearningRate)

	inputHiddenDeltas := hiddenGradients.
		DotProduct(inputs.Transpose())

	n.InputHiddenWeights = n.InputHiddenWeights.
		AddMatrix(inputHiddenDeltas)
	n.HiddenBias = n.HiddenBias.
		AddMatrix(hiddenGradients)
}
