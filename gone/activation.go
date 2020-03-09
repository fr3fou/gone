package gone

type Activation struct {
	F      func(float64) float64
	FPrime func(float64) float64
}
