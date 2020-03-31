package gone

import "github.com/fr3fou/gone/matrix"

// Optimizer is the optimizer type
type Optimizer func(n *NeuralNetwork, dataSet DataSet)

// SGD is Stochastic Gradient Descent (On-line Training)
func SGD() Optimizer {
	// OK THIS WORKS
	return func(n *NeuralNetwork, dataSet DataSet) {
		MGBD(1)(n, dataSet)
	}
}

// MBGD is a Mini-Batch Gradient Descent (Batch training)
func MGBD(batchSize int) Optimizer {
	// TODO: fix to work with any batchSize (currently works only with 1 - SGD)
	return func(n *NeuralNetwork, dataSet DataSet) {
		for i := 0; i < len(dataSet); i++ {
			batch := dataSet.Batch(i, batchSize)
			batch.Shuffle()

			lenLayers := len(n.Layers)
			lenWeights := lenLayers - 1

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

					n.Weights[i] = n.Weights[i].AddMatrix(currentDeltas)
					n.Biases[i] = n.Biases[i].AddMatrix(currentGradients)
				}
			}
		}
	}
}

// GD is a normal gradient descent (Optimizes after the entire data set)
func GD() Optimizer {
	// TODO: should work after fixing normal MBGD
	return func(n *NeuralNetwork, dataSet DataSet) {
		MGBD(len(dataSet))(n, dataSet)
	}
}
