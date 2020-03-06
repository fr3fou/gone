package main

import (
	"math/rand"
)

const Inputs = 2

type Perceptron struct {
	XWeight      float64
	YWeight      float64
	LearningRate float64
	Epochs       int
}

func NewPerceptron(lRate float64) *Perceptron {
	xW := float64(rand.Intn(2) - 1)
	yW := float64(rand.Intn(2) - 1)

	return &Perceptron{
		XWeight:      xW,
		YWeight:      yW,
		LearningRate: lRate,
	}
}

func (p *Perceptron) Guess(input Point) int {
	sum := 0.0

	sum += input.X * p.XWeight
	sum += input.Y * p.YWeight

	return Sign(sum)
}

func (p *Perceptron) Train(inputs []Point) {
	for i := 0; i < p.Epochs; i++ {
		for _, input := range inputs {
			guess := p.Guess(input)
			err := float64(input.Label - guess)

			p.XWeight += err * p.LearningRate
			p.YWeight += err * p.LearningRate
		}
	}
}

// Sign is an activation function
func Sign(n float64) int {
	if n >= 0 {
		return 1
	}

	return -1
}
