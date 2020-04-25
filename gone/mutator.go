package gone

import (
	stdrand "math/rand"

	"github.com/fr3fou/gone/rand"
)

// Mutator is a function for mutating genes
type Mutator func(val float64) float64

// GaussianMutation applies a randomly distributed gaussian mutation using mutationRate which should be a number in the range [0.0, 1.0] and represents a probability for a mutation to occur
func GaussianMutation(mutationRate float64, stdenv, mean float64) Mutator {
	return func(val float64) float64 {
		if rand.Float(0, 1) >= mutationRate {
			mutation := stdrand.NormFloat64()*stdenv + mean // stdev = 0.5
			return val + mutation
		}

		return val
	}
}
