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
	x := rand.Float64() * Width
	y := rand.Float64() * Height
	label := -1

	if x > y {
		label = 1
	} else {
		label = -1
	}

	return &Point{
		X:     x,
		Y:     y,
		Label: label,
	}
}
