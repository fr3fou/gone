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
	p := NewPerceptron()
	fmt.Println(p.Guess([Inputs]float64{-1, 0.5}))
}
