package gone

import "math"

// Activation is an activation function
// it contains the normal f(x) and the derivative f'(x)
type Activation struct {
	Name   acitvationName
	F      func(x float64) float64
	FPrime func(y float64) float64
}

type acitvationName string

const (
	sigmoid acitvationName = "sigmoid"
	relu    acitvationName = "relu"
	softmax acitvationName = "softmax"
	id      acitvationName = "id"
)

// Sigmoid is a sigmoid activation function
func Sigmoid() Activation {
	return Activation{
		Name: sigmoid,
		F: func(x float64) float64 {
			return 1 / (1 + math.Exp(-x))
		},
		FPrime: func(y float64) float64 {
			return y * (1 - y)
		},
	}
}

// ReLU is a ReLU activation function
func ReLU() Activation {
	return Activation{
		Name: relu,
		F: func(x float64) float64 {
			return math.Max(0, x)
		},
		FPrime: func(x float64) float64 {
			if x > 0 {
				return 1
			}
			return 0
		},
	}
}

// Identity is the identity (linear) function
// f(x) = x
func Identity() Activation {
	return Activation{
		Name: id,
		F: func(x float64) float64 {
			return x
		},
		FPrime: func(_ float64) float64 {
			return 1
		},
	}
}
