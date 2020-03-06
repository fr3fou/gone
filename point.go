package main

import "math/rand"

const (
	Width  = 800
	Height = 800
)

type Point struct {
	X     float64
	Y     float64
	Label int
}

func NewPoint() *Point {
	x := float64(rand.Intn(2) - 1) // range between -1 and 1
	y := float64(rand.Intn(2) - 1) // range between -1 and 1
	label := -1

	if x > y { // f(x) = x
		label = 1
	}

	return &Point{
		X:     x,
		Y:     y,
		Label: label,
	}
}

func f(x float64) float64 {
	return 0.3*x + 0.2
}

func NewPointXY(x, y float64) *Point {
	label := -1
	if y > f(x) { // f(x) = x
		label = 1
	}

	return &Point{
		X:     x,
		Y:     y,
		Label: label,
	}
}
