package gone

import "github.com/fr3fou/matrigo"

// Optimizer is the optimizer type (returns the error).
type Optimizer func(n *NeuralNetwork, dataSet DataSet) float64

// SGD is Stochastic Gradient Descent (On-line Training)
func SGD() Optimizer {
	return func(n *NeuralNetwork, dataSet DataSet) float64 {
		return MBGD(1)(n, dataSet)
	}
}

// MBGD is a Mini-Batch Gradient Descent (Batch training)
func MBGD(batchSize int) Optimizer {
	return func(n *NeuralNetwork, dataSet DataSet) float64 {
		err := 0.0

		for i := 0; i < len(dataSet); i++ {
			batch := dataSet.Batch(i, batchSize)

			lenLayers := len(n.Layers)
			lenWeights := lenLayers - 1

			deltas := make([]matrigo.Matrix, lenWeights)
			gradients := make([]matrigo.Matrix, lenWeights)

			// Zero the matrices
			for i := 0; i < lenWeights; i++ {
				deltas[i] = matrigo.
					New(n.Weights[i].Rows, n.Weights[i].Columns, nil)
				gradients[i] = matrigo.
					New(n.Biases[i].Rows, n.Biases[i].Columns, nil)
			}

			for _, ds := range batch {
				inputs := matrigo.NewFromArray(ds.Inputs)
				targets := matrigo.NewFromArray(ds.Targets)
				outputs := n.predict(inputs)

				err += n.Loss.F(outputs, targets)
				loss := n.Loss.FPrime(outputs, targets)

				var currentGradients matrigo.Matrix
				var currentDeltas matrigo.Matrix

				for i := lenWeights - 1; i >= 0; i-- {
					// Ignore the first time
					if i < lenWeights-1 {
						loss = n.Weights[i+1].Transpose().Multiply(loss)
					}

					currentGradients = n.Activations[i+1].
						Map(func(val float64, x, y int) float64 {
							return n.Layers[i+1].Activator.FPrime(val)
						}).
						HadamardProduct(loss).
						Scale(n.LearningRate)

					currentDeltas = currentGradients.
						Multiply(n.Activations[i].Transpose())

					deltas[i] = deltas[i].AddMatrix(currentDeltas)
					gradients[i] = gradients[i].AddMatrix(currentGradients)
				}
			}

			for i := 0; i < lenWeights; i++ {
				deltas[i] = deltas[i].Map(
					func(val float64, x, y int) float64 {
						return val / float64(batchSize) // average the changes
					},
				)

				gradients[i] = gradients[i].Map(
					func(val float64, x, y int) float64 {
						return val / float64(batchSize) // average the changes
					},
				)

				n.Weights[i] = n.Weights[i].AddMatrix(deltas[i])
				n.Biases[i] = n.Biases[i].AddMatrix(gradients[i])
			}
		}

		return err
	}
}

// GD is a normal gradient descent (Optimizes after the entire data set)
func GD() Optimizer {
	return func(n *NeuralNetwork, dataSet DataSet) float64 {
		return MBGD(len(dataSet))(n, dataSet)
	}
}
