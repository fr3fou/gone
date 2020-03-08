package main

import (
	"fmt"
	"math/rand"
	"time"
)

func init() {
	rand.Seed(time.Now().UnixNano())
}

func main() {
	p := NewPerceptron(0.1, 10000)

	// Training data
	pts := []Point{}
	for i := 0; i < 1000; i++ {
		pt := Point{
			X: float64(rand.Int31n(201) - 101),
			Y: float64(rand.Int31n(201) - 101),
		}
		pt.Label = aboveF(pt.X, pt.Y)
		pts = append(pts, pt)
	}

	p.Train(pts)
	fmt.Printf("%d%% correct.\n", p.Verify())
}
