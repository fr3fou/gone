package main

import "math/rand"

func randFloat(min, max float64) float64 {
	return min + rand.Float64()*(max-min)
}
