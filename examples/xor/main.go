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
		0.1,
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

	g.ToggleDebug(true)

	g.Train(
		gone.SGD(),
		gone.DataSet{
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
		2000,
	)

	if err := g.Save("xor.gone"); err != nil {
		panic(err)
	}

	fmt.Println("1 0 -> ", g.Predict([]float64{1, 0}), "should've been around 1")
	fmt.Println("0 1 -> ", g.Predict([]float64{0, 1}), "should've been around 1")
	fmt.Println("1 1 -> ", g.Predict([]float64{1, 1}), "should've been around 0")
	fmt.Println("0 0 -> ", g.Predict([]float64{0, 0}), "should've been around 0")
}
