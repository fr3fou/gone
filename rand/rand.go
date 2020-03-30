package rand

import "math/rand"

func Float(min, max float64) float64 {
	return rand.Float64()*max + min
}
