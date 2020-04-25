package gone

import (
	stdrand "math/rand"

	"github.com/fr3fou/gone/rand"
)

// Mutator is a function for mutating genes
type Mutator func(val float64) float64

// GaussianMutation applies a randomly distributed gaussian mutation
// mutationRate should be a number in the range [0.0, 1.0] and represents a probability
func GaussianMutation(mutationRate float64) Mutator {
	return func(val float64) float64 {
		if rand.Float(0, 1) >= mutationRate {
			mutation := stdrand.NormFloat64() * 0.5 // stdev = 0.5
			return val + mutation
		}

		return val
	}
}
