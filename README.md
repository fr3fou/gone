# gone

A simple neural network library in Go from scratch. 0 dependencies.

## Example

```go
	g := gone.New(
		.1,
		100,
		gone.Classification,
		// gone.MSE,
		gone.Layer{
			Nodes: 2,
		},
		gone.Layer{
			Nodes:     3,
			Activator: gone.ReLU(),
		},
		gone.Layer{
			Nodes:     3,
			Activator: gone.ReLU(),
		},
		gone.Layer{
			Nodes: 1,
			// we shouldn't use ReLU on the outputs, so we fallback to Id
		},
	)

	g.Predict([]float64{1, 2})
```

## TODO

### `gone/`

- [ ] Types of task:
  - [ ] Classification
  - [ ] Regression
- [x] Bias
  - [x] Matrix, rather than a single number
- [x] Feedforward (Predict)
- [ ] Train
  - [x] Support shuffling the data
  - [x] Epochs
  - [x] Backpropagation
  - [ ] Batching
  - [ ] Different loss functions
    - [ ] Mean Squared Error
    - [ ] Cross Entropy Error
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
- [x] Partial Derivatives ~
- [ ] Linear vs non-linear problems (activation function)
- [x] Gradient Descent
  - [x] (Batch) Gradient Descent (GD)
  - [x] Stochastic Gradient Descent (SGD)
  - [x] Mini-Batch Gradient Descent (MBGD?)
- [ ] Softmax (needed for multi class classification!)
- [ ] Mean Squared Error
- [ ] Cross Entropy Error (needed for multi class classification!)
- [ ] How to determine how many layers and nodes to use

### Examples

- [x] XOR Problem
- [ ] Digit Classifier

### Shoutouts

- [Davis Josephs](https://github.com/josephsdavid) - was of HUGE help with algebra and other ML-related questions; also helped me spot some nasty bugs!

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
- https://arxiv.org/pdf/1802.01528.pdf
- https://github.com/stevenmiller888/mind/blob/master/index.js
- https://github.com/stevenmiller888/go-mind
- https://medium.com/yottabytes/everything-you-need-to-know-about-gradient-descent-applied-to-neural-networks-d70f85e0cc14
- https://towardsdatascience.com/deep-learning-which-loss-and-activation-functions-should-i-use-ac02f1c56aa8
- https://github.com/milosgajdos/go-neural
- https://ml-cheatsheet.readthedocs.io/en/latest/forwardpropagation.html
- https://medium.com/coinmonks/representing-neural-network-with-vectors-and-matrices-c6b0e64db9fb
