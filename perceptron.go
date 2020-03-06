package main

import (
	"math/rand"
)

const Inputs = 2

type Perceptron struct {
	Weights [2]float64
}

func NewPerceptron() *Perceptron {
	weights := [Inputs]float64{}
	for i := 0; i < Inputs; i++ {
		weights[i] = float64(rand.Intn(2) - 1)
	}

	return &Perceptron{
		Weights: weights,
	}
}

func (p *Perceptron) Guess(inputs [Inputs]float64) int {
	sum := 0.0

	for i := range inputs {
		sum += inputs[i] * p.Weights[i]
	}

	return Sign(sum)
}

// Sign is an activation function
func Sign(n float64) int {
	if n >= 0 {
		return 1
	}

	return -1
}
