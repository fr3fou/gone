package main

import (
	"log"
	"math"
	"math/rand"
)

const Inputs = 2

// Perceptron is a perceptron
type Perceptron struct {
	XWeight      float64
	YWeight      float64
	Bias         float64
	LearningRate float64
	Epochs       int
}

// NewPerceptron is a constructor for a Perceptron
func NewPerceptron(lRate float64, epochs int) *Perceptron {
	xW := rand.Float64() - 1 // range between -1 and 1
	yW := rand.Float64() - 1 // range between -1 and 1

	return &Perceptron{
		XWeight:      xW,
		YWeight:      yW,
		Epochs:       epochs,
		LearningRate: lRate,
	}
}

// Feedfoward predicts based on the input
func (p *Perceptron) Feedfoward(input Point) int {
	sum := 0.0

	sum += input.X * p.XWeight
	sum += input.Y * p.YWeight
	sum += input.Bias * p.Bias

	return Sign(sum)
}

// Train trains the perceptron based on inputs
func (p *Perceptron) Train(inputs []Point) {
	bestErr := math.MaxFloat64
	lastErr := 0.0
	for i := 0; i < p.Epochs; i++ {
		for _, input := range inputs {
			guess := p.Feedfoward(input)
			err := float64(input.Label - guess)

			bestErr = math.Min(err, bestErr)
			lastErr = err

			p.XWeight += err * p.LearningRate
			p.YWeight += err * p.LearningRate
			p.Bias += err * p.LearningRate
		}
	}

	log.Printf("Training completed with error %f and best error %f", lastErr, bestErr)
	log.Printf("%+v", p)
}

// Sign is an activation function
func Sign(n float64) int {
	if n >= 0 {
		return 1
	}

	return -1
}
