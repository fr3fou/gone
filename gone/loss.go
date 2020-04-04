package gone

import "github.com/fr3fou/gone/matrix"

// Loss is a loss function
// it contains the normal f(x) and the derivative f'(x)
type Loss struct {
	Name   lossName
	F      func(y, yHat matrix.Matrix) float64
	FPrime func(y, yHat matrix.Matrix) matrix.Matrix
}

type lossName string

const (
	mse lossName = "mse"
)

func getLossFromname(a lossName) Loss {
	switch a {
	case mse:
		return MSE()
	default:
		return MSE()
	}
}

// MSE is the Mean Squared Error
// NOT IMPLEMENTED YET
func MSE() Loss {
	return Loss{
		Name: mse,
		F: func(y, yHat matrix.Matrix) float64 {
			out := y.SubtractMatrix(yHat)
			out = out.HadamardProduct(out)
			return out.Sum() / float64(y.Rows)
		},
		FPrime: func(y, yHat matrix.Matrix) matrix.Matrix {
			panic("todo")
		},
	}
}
