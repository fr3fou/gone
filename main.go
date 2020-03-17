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
		0.1, // learning rate (alpha)
		// 1,   // batch size
		gone.Classification,
		// gone.MSE,
		gone.Layer{
			Nodes: 2,
		},
		gone.Layer{
			Nodes:     2,
			Activator: gone.Sigmoid(),
		},
		gone.Layer{
			Nodes: 1,
			// we shouldn't use ReLU on the outputs, so we fallback to Id
		},
	)

	fmt.Println("Training...")

	g.Train(gone.DataSet{
		{
			Inputs:  []float64{1, 0},
			Targets: []float64{1},
		},
		{
			Inputs:  []float64{0, 1},
			Targets: []float64{1},
		},
		{
			Inputs:  []float64{1, 1},
			Targets: []float64{0},
		},
		{
			Inputs:  []float64{0, 0},
			Targets: []float64{0},
		},
	},
		1000,
	)

	fmt.Println("Finished...")
	fmt.Println("1 0 -> ", g.Predict([]float64{1, 0}), "should've been around 1")
	fmt.Println("0 1 -> ", g.Predict([]float64{0, 1}), "should've been around 1")
	fmt.Println("1 1 -> ", g.Predict([]float64{1, 1}), "should've been around 0")
	fmt.Println("0 0 -> ", g.Predict([]float64{0, 0}), "should've been around 0")
}

func _main() {
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
