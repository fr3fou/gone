package main

import (
	"fmt"
	stdrand "math/rand"
	"time"

	"github.com/fr3fou/gone/gone"
	"github.com/fr3fou/gone/perceptron"
	"github.com/fr3fou/gone/point"
	"github.com/fr3fou/gone/rand"
)

func init() {
	stdrand.Seed(time.Now().UnixNano())
}

func main() {
	g := gone.New(
		.1,
		gone.Classification,
		gone.Layer{
			Nodes: 2,
		},
		gone.Layer{
			Nodes:     3,
			Activator: gone.ReLU,
		},
		gone.Layer{
			Nodes:     3,
			Activator: gone.ReLU,
		},
		gone.Layer{
			Nodes: 1,
			// we shouldn't use ReLU on the outputs, so we fallback to Id
		},
	)

	g.Train(gone.TrainData{
		{
			Inputs: []float64{1, 0},
			Labels: []float64{1},
		},
		{
			Inputs: []float64{0, 1},
			Labels: []float64{1},
		},
		{
			Inputs: []float64{1, 1},
			Labels: []float64{0},
		},
		{
			Inputs: []float64{0, 0},
			Labels: []float64{0},
		},
	},
		500,
	)
}

func main2() {
	p := perceptron.New(0.1, 10000)

	// Training data
	pts := []point.Point{}
	for i := 0; i < 1000; i++ {
		pt := point.Point{
			X: rand.Float(-100, 101),
			Y: rand.Float(-100, 101),
		}
		pt.Label = point.AboveF(pt.X, pt.Y, f)
		pts = append(pts, pt)
	}

	p.Train(pts)
	fmt.Printf("%d%% correct.\n", p.Verify(f))
}

func f(x float64) float64 {
	return 3*x + 2
}
