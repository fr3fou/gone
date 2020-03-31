package rand

import "math/rand"

// Float returns a random float in the range (min, max]
func Float(min, max float64) float64 {
	return rand.Float64()*max + min
}
