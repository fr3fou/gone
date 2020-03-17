package gone

import "math"

// Activation is an activation function
// it contains the normal f(x) and the derivative f'(x)
type Activation struct {
	F      func(x float64) float64
	FPrime func(x float64) float64
}

// Sigmoid is a sigmoid activation function
func Sigmoid() Activation {
	return Activation{
		F: func(x float64) float64 {
			return 1 / (1 + math.Exp(-x))
		},
		FPrime: func(x float64) float64 {
			return x * (1 - x)
		},
	}
}

// ReLU is a ReLU activation function
func ReLU() Activation {
	return Activation{
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
		F: func(x float64) float64 {
			return x
		},
		FPrime: func(_ float64) float64 {
			return 1
		},
	}
}
