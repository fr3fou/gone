# gone

A simple neural network library in Go from scratch. 0 dependencies.

## Example

```go
	g := gone.New(
		.1,
		100,
		gone.Classification,
		gone.Layer{
			Nodes: 2,
		},
		gone.Layer{
			Nodes:     3,
			Activator: gone.ReLU,
		},
		gone.Layer{
			Nodes:     3,
			Activator: gone.ReLU,
		},
		gone.Layer{
			Nodes: 1,
			// we shouldn't use ReLU on the outputs, so we fallback to Id
		},
	)

	g.Predict([]float64{1, 2})
```

## TODO:

### `gone/`

- [ ] Types of task:
  - [ ] Classification
  - [ ] Regression
- [x] Bias
- [x] Feedforward (Predict)
- [ ] Train
  - [x] Support shuffling the data
  - [x] Epochs
  - [ ] Backpropagation
  - [x] Batching
- [ ] Saving data
- [ ] Loading data

### `matrix/`

- [x] Randomize
- [x] Transpose
- [x] Scale
- [x] AddMatrix
- [x] Add
- [x] SubtractMatrix
- [x] Subtract
- [x] Multiply
- [x] Multiply
- [x] Flatten
- [x] NewFromArray - makes a single row
- [x] Map
- [x] Methods to support chaining

```go
	    n.Weights[i].
		Multiply(output).                         // weighted sum of the previous layer)
		Add(n.Layers[i+1].Bias).                  // bias
		Map(func(val float64, x, y int) float64 { // activation
			return n.Layers[i+1].Activator.F(val)
		})
```

### Research

- [x] Derivatives ~
- [ ] Linear vs non-linear problems (activation function)
- [x] Gradient Descent ~
  - [x] Stochastic vs Batching ~
- [ ] Softmax (needed for multi class classification!)
- [ ] Mean Squared Error
- [ ] Cross Entropy Error (needed for multi class classification!)
- [ ] How to determine how many layers and nodes to use

### Examples

- [ ] XOR Problem
- [ ] Digit Classifier

## Resources used:

- https://www.analyticsvidhya.com/blog/2020/01/fundamentals-deep-learning-activation-functions-when-to-use-them/
- https://www.youtube.com/watch?v=XJ7HLz9VYz0&list=PLRqwX-V7Uu6Y7MdSCaIfsxc561QI0U0Tb
- https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi
- http://matrixmultiplication.xyz/
- https://www.khanacademy.org/math/precalculus/x9e81a4f98389efdf:matrices/x9e81a4f98389efdf:properties-of-matrix-addition-and-scalar-multiplication/a/properties-of-matrix-addition
- https://www.wikiwand.com/en/Matrix_(mathematics)
- https://www.wikiwand.com/en/Activation_function
- https://www.jeremyjordan.me/intro-to-neural-networks/
- https://www.arxiv-vanity.com/papers/2003.02139/
- https://machinelearningmastery.com/gentle-introduction-mini-batch-gradient-descent-configure-batch-size/
- http://neuralnetworksanddeeplearning.com/chap2.html
