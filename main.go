package main

import (
	"fmt"
	stdrand "math/rand"
	"time"

	"github.com/fr3fou/gone/matrix"
	"github.com/fr3fou/gone/perceptron"
	"github.com/fr3fou/gone/point"
	"github.com/fr3fou/gone/rand"
)

func init() {
	stdrand.Seed(time.Now().UnixNano())
}

func main() {
	p := perceptron.New(0.1, 10000)

	m := matrix.New(5, 3, nil)
	m.Randomize()
	fmt.Println(m)

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
