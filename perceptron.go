package main

import (
	"log"
	"math"
	"math/rand"
)

const Inputs = 2

type Perceptron struct {
	XWeight      float64
	YWeight      float64
	LearningRate float64
	Epochs       int
}

func NewPerceptron(lRate float64, epochs int) *Perceptron {
	xW := float64(rand.Intn(2) - 1)
	yW := float64(rand.Intn(2) - 1)

	return &Perceptron{
		XWeight:      xW,
		YWeight:      yW,
		Epochs:       epochs,
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
	bestErr := math.MaxFloat64
	lastErr := 0.0
	for i := 0; i < p.Epochs; i++ {
		for _, input := range inputs {
			guess := p.Guess(input)
			err := float64(input.Label - guess)

			bestErr = math.Min(err, bestErr)
			lastErr = err

			p.XWeight += err * p.LearningRate
			p.YWeight += err * p.LearningRate
		}
	}

	log.Printf("Training completed with error %f and best error %f", lastErr, bestErr)
}

// Sign is an activation function
func Sign(n float64) int {
	if n >= 0 {
		return 1
	}

	return -1
}
