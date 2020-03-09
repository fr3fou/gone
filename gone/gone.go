package gone

import (
	"github.com/fr3fou/gone/matrix"
)

type Activation func(float64) float64

type Layer struct {
	Nodes int
}

type NeuralNetwork struct {
	Weights            []matrix.Matrix
	Errors             []matrix.Matrix
	LearningRate       float64
	ActivationFunction Activation
}

func New(lr float64, layers ...Layer) *NeuralNetwork {
}
