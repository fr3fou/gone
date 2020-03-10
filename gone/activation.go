package gone

import "math"

type Activation struct {
	F      func(float64) float64
	FPrime func(float64) float64
}

var Sigmoid = Activation{
	F:      sigmoid,
	FPrime: sigmoidPrime,
}

func sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func sigmoidPrime(x float64) float64 {
	return x * (1 - x)
}

var ReLU = Activation{
	F:      relu,
	FPrime: reluPrime,
}

func relu(x float64) float64 {
	return math.Max(0, x)
}

func reluPrime(x float64) float64 {
	if x > 0 {
		return 1
	}

	return 0
}

var Id = Activation{
	F:      id,
	FPrime: idPrime,
}

func id(x float64) float64 {
	return x
}

func idPrime(_ float64) float64 {
	return 1
}
