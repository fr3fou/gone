package perceptron

import (
	"github.com/fr3fou/gone/point"
	"github.com/fr3fou/gone/rand"
)

// Perceptron is a perceptron
type Perceptron struct {
	XWeight      float64
	YWeight      float64
	Bias         float64
	LearningRate float64
	Epochs       int
}

// New is a constructor for a Perceptron
func New(lRate float64, epochs int) *Perceptron {
	xW := rand.Float(-1, 2)
	yW := rand.Float(-1, 2)

	return &Perceptron{
		XWeight:      xW,
		YWeight:      yW,
		Epochs:       epochs,
		LearningRate: lRate,
	}
}

// Feedfoward predicts based on the input
func (p *Perceptron) Feedfoward(input point.Point) int {
	sum := 0.0

	sum += input.X * p.XWeight
	sum += input.Y * p.YWeight
	sum += input.Bias * p.Bias

	return Sign(sum)
}

// Train trains the perceptron based on inputs
func (p *Perceptron) Train(inputs []point.Point) {
	for i := 0; i < p.Epochs; i++ {
		for _, input := range inputs {
			guess := p.Feedfoward(input)
			err := float64(input.Label - guess)

			p.XWeight += err * p.LearningRate * input.X
			p.YWeight += err * p.LearningRate * input.Y
			p.Bias += err * p.LearningRate
		}
	}
}

// Verify benches the perceptron
func (p *Perceptron) Verify(f point.F) int {
	correct := 0
	for i := 0; i < 100; i++ {
		pt := point.Point{
			X: rand.Float(-100, 101),
			Y: rand.Float(-100, 101),
		}
		pt.Label = point.AboveF(pt.X, pt.Y, f)
		res := p.Feedfoward(pt)
		if res == pt.Label {
			correct++
		}
	}
	return correct
}

// Sign is an activation function
func Sign(n float64) int {
	if n >= 0 {
		return 1
	}

	return -1
}
