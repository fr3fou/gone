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
	p := NewPerceptron(0.1, 1000)

	// Training data
	pts := []Point{}
	for i := 0; i < 1000; i++ {
		x := randFloat(-100, 100)
		y := randFloat(-100, 100)
		pts = append(pts, *NewPointXY(x, y))
	}

	p.Train(pts)
	fmt.Printf("%d%% correct.\n", p.Verify())
}
