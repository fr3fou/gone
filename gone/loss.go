package gone

import "github.com/fr3fou/gone/matrix"

type Loss struct {
	F func(outputs, targets matrix.Matrix) matrix.Matrix
	FPrime func ??? 
}

func MSE(outputs, targets matrix.Matrix) matrix.Matrix {
	// Calculate the error
	// Error_{i,j} = Target_{i,j} - Outputs_{i,j}
	errs := targets.SubtractMatrix(outputs)

	// Raise them to the 2nd power
	squared := errs.HadamardProduct(errs)

	// Make a new one dimensional vector of all the mean errors
	return matrix.Map(matrix.New(squared.Rows, 1, nil), func(val float64, x, y int) float64 {
		sum := 0.0

		for i := 0; i < squared.Rows; i++ {
			sum += squared.Data[i][x]
		}

		return sum / float64(squared.Rows)
	})

}
