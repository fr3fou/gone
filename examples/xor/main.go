package main

import (
	"fmt"
	stdrand "math/rand"
	"time"

	"github.com/fr3fou/gone/gone"
)

func init() {
	stdrand.Seed(time.Now().UnixNano())
}

func main() {
	g := gone.New(
		0.1, // learning rate (alpha)
		// 1,   // batch size
		// gone.Classification,
		gone.SGD(),
		// gone.MSE,
		gone.Layer{
			Nodes: 2,
		},
		gone.Layer{
			Nodes:     4,
			Activator: gone.Sigmoid(),
		},
		gone.Layer{
			Nodes: 1,
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
		5000,
	)

	fmt.Println("Finished...")
	fmt.Println("1 0 -> ", g.Predict([]float64{1, 0}), "should've been around 1")
	fmt.Println("0 1 -> ", g.Predict([]float64{0, 1}), "should've been around 1")
	fmt.Println("1 1 -> ", g.Predict([]float64{1, 1}), "should've been around 0")
	fmt.Println("0 0 -> ", g.Predict([]float64{0, 0}), "should've been around 0")
}
