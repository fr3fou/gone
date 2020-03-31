package gone

import "github.com/fr3fou/gone/matrix"

// Optimizer is the optimizer type
type Optimizer func(n *NeuralNetwork, dataSet DataSet)

// SGD is Stochastic Gradient Descent (On-line Training)
func SGD() Optimizer {
	return func(n *NeuralNetwork, dataSet DataSet) {
		MGBD(1)(n, dataSet)
	}
}

// MBGD is a Mini-Batch Gradient Descent (Batch training)
func MGBD(batchSize int) Optimizer {
	return func(n *NeuralNetwork, dataSet DataSet) {
		for i := 0; i < len(dataSet); i++ {
			batch := dataSet.Batch(i, batchSize)
			batch.Shuffle()

			lenLayers := len(n.Layers)
			lenWeights := lenLayers - 1

			deltas := make([]matrix.Matrix, lenWeights)
			gradients := make([]matrix.Matrix, lenWeights)

			// Zero the matrices
			for i := 0; i < lenWeights; i++ {
				deltas[i] = matrix.
					New(n.Weights[i].Rows, n.Weights[i].Columns, nil)
				gradients[i] = matrix.
					New(n.Biases[i].Rows, n.Biases[i].Columns, nil)
			}

			for _, ds := range batch {
				inputs := matrix.NewFromArray(ds.Inputs)
				targets := matrix.NewFromArray(ds.Targets)
				outputs := n.predict(inputs)

				// TODO: custom loss
				err := targets.SubtractMatrix(outputs)

				var currentGradients matrix.Matrix
				var currentDeltas matrix.Matrix

				for i := lenWeights - 1; i >= 0; i-- {
					// Ignore the first time
					if i < lenWeights-1 {
						err = n.Weights[i+1].Transpose().DotProduct(err)
					}

					currentGradients = n.Activations[i+1].
						Map(func(val float64, x, y int) float64 {
							return n.Layers[i+1].Activator.FPrime(val)
						}).
						HadamardProduct(err).
						Scale(n.LearningRate)

					currentDeltas = currentGradients.
						DotProduct(n.Activations[i].Transpose())

					deltas[i] = deltas[i].AddMatrix(currentDeltas)
					gradients[i] = gradients[i].AddMatrix(currentGradients)
				}
			}

			for i := 0; i < lenWeights; i++ {
				deltas[i] = deltas[i].Map(
					func(val float64, x, y int) float64 {
						return val / float64(batchSize)
					},
				)

				gradients[i] = gradients[i].Map(
					func(val float64, x, y int) float64 {
						return val / float64(batchSize)
					},
				)

				n.Weights[i] = n.Weights[i].AddMatrix(deltas[i])
				n.Biases[i] = n.Biases[i].AddMatrix(gradients[i])
			}
		}
	}
}

// GD is a normal gradient descent (Optimizes after the entire data set)
func GD() Optimizer {
	return func(n *NeuralNetwork, dataSet DataSet) {
		MGBD(len(dataSet))(n, dataSet)
	}
}
