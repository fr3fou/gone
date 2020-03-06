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
	p := NewPerceptron(0.1, 20)

	// Training data
	pts := []Point{}
	for i := 0; i < 100; i++ {
		pts = append(pts, *NewPointXY(float64(i), f(float64(i))))
	}

	p.Train(pts)
	fmt.Println(p.Feedfoward(Point{X: 0, Y: f(0)}))
}
