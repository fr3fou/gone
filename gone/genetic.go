package gone

import (
	"errors"
	"math/rand"
)

var (
	// ErrWeightsNotMatch is an error for when the parents don't have the same amount of weights.
	ErrWeightsNotMatch = errors.New("gone: parents must have the exact same amount of weights")
)

// Crossover applies a crossover between 2 neural networks by getting random bits of both to create a child
func (firstParent *NeuralNetwork) Crossover(secondParent *NeuralNetwork) (*NeuralNetwork, error) {
	lenWeights := len(firstParent.Weights)
	if lenWeights != len(secondParent.Weights) {
		return nil, ErrWeightsNotMatch
	}

	child := New(firstParent.LearningRate, firstParent.Loss, firstParent.Layers...)

	// [0, 5) ->
	// [0, 4] ->
	// 3 ->
	// first parent gives until 3rd layer ->
	// 5 - 3 = 2 ->
	// second parent gives from 2nd layer until end

	l1 := rand.Intn(lenWeights)
	l2 := lenWeights - l1

	for i := 0; i < l1; i++ {
		child.Weights[i] = firstParent.Weights[i].Copy() // Copy the first l1 amount of the weights
		child.Biases[i] = firstParent.Biases[i].Copy()   // Copy the first l1 amount of the biases
	}

	for i := l2; i < lenWeights; i++ {
		child.Weights[i] = secondParent.Weights[i].Copy() // Copy the second l2 amount of the weights
		child.Biases[i] = secondParent.Biases[i].Copy()   // Copy the second l2 amount of the biases
	}

	return child, nil
}

// Mutate randomly mutates some weights and biases
func (n *NeuralNetwork) Mutate(mutator Mutator) {
	lenWeights := len(n.Weights)

	for i := 0; i < lenWeights; i++ {
		n.Weights[i] = n.Weights[i].
			Map(func(val float64, x, y int) float64 {
				return mutator(val)
			})

		n.Biases[i] = n.Biases[i].
			Map(func(val float64, x, y int) float64 {
				return mutator(val)
			})
	}
}
