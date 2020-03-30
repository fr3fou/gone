package gone

import "github.com/fr3fou/gone/matrix"

// Optimizer is the optimizer type
type Optimizer func(n *NeuralNetwork, dataSet DataSet, epochs int)

func SGD() Optimizer {
	return func(n *NeuralNetwork, dataSet DataSet, epochs int) {
		for epoch := 0; epoch < epochs; epoch++ {
			dataSet.Shuffle()
			for _, ds := range dataSet {
				// Stochastic Gradient Descent (On-line Training)
				inputs := matrix.NewFromArray(ds.Inputs)
				targets := matrix.NewFromArray(ds.Targets)
				outputs := n.predict(inputs)

				lenLayers := len(n.Layers)
				lenWeights := lenLayers - 1 // always one less than the layers (we don't have weights for the inputs)

				err := targets.SubtractMatrix(outputs)

				var gradients matrix.Matrix
				var deltas matrix.Matrix

				for i := lenWeights - 1; i >= 0; i-- {
					// The next error is equal to the current error multiplied
					// by the previous weight matrix but transposed!
					// The outputs of the previous layer must match with the inputs
					// of the current layer (the current layer's errors have that shape)
					//
					// We need not compute the error the first time as it was done
					// outside the loop
					if i < lenWeights-1 {
						err = n.Weights[i+1].Transpose().DotProduct(err)
					}

					gradients = n.Activations[i+1].
						Map(func(val float64, x, y int) float64 {
							return n.Layers[i+1].Activator.FPrime(val)
						}).
						HadamardProduct(err).
						Scale(n.LearningRate)

					deltas = gradients.
						DotProduct(n.Activations[i].Transpose())
						// Scale(n.LearningRate)

					n.Weights[i] = n.Weights[i].AddMatrix(deltas)
					n.Biases[i] = n.Biases[i].AddMatrix(gradients)
				}
			}
		}
	}
}
