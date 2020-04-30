# gone

[![Github Actions Widget]][github actions] [![GoReport Widget]][goreport] [![GoDoc Widget]][godoc]

A simple neural network library in Go from scratch. 0 dependencies\*

_there are 0 neural network related dependencies, the only dependencies are for testing ([stretchr/testify](github.com/stretchr/testify)) and for persistence ([golang/protobuf](github.com/golang/protobuf))_

[goreport widget]: https://goreportcard.com/badge/github.com/fr3fou/gone
[goreport]: https://goreportcard.com/report/github.com/fr3fou/gone
[github actions widget]: https://github.com/fr3fou/gone/workflows/Test/badge.svg
[github actions]: https://github.com/fr3fou/gone/actions
[godoc]: http://pkg.go.dev/github.com/fr3fou/gone
[godoc widget]: https://godoc.org/github.com/fr3fou/gone?status.svg

## Example

### Getting started

```go
 g := gone.New(
    0.1,
    gone.MSE(),
    gone.Layer{
      Nodes: 2,
    },
    gone.Layer{
      Nodes:     4,
      Activator: gone.Sigmoid(),
    },
    gone.Layer{
      Nodes: 1,
    },
 )

 g.Train(gone.SGD(), gone.DataSet{
  {
    Inputs:  []float64{1, 0},
    Targets: []float64{1},
  },
  {
    Inputs:  []float64{0, 1},
    Targets: []float64{1},
  },
  {
    Inputs:  []float64{1, 1},
    Targets: []float64{0},
  },
  {
    Inputs:  []float64{0, 0},
    Targets: []float64{0},
  },
 }, 5000)

 g.Predict([]float64{1, 1})
```

### Saving model to disk

```go
 g.Save("test.gone")

```

### Loading model back into memory

```go
 g, err := gone.Load("test.gone")
```

## TODO

### `gone/`

- [x] Types of task:
  - [x] Classification - `softmax` (soon to be implemented) as the last layer's activation function
  - [x] Regression - `sigmoid` as the last layer's activation function
- [x] Bias
  - [x] Matrix, rather than a single number
- [x] Feedforward (Predict)
- [ ] Train
  - [x] Support shuffling the data
  - [x] Epochs
  - [x] Backpropagation
  - [x] Batching
  - [ ] Different loss functions
    - [x] Mean Squared Error
    - [ ] Cross Entropy Error
- [x] Saving data - Done thanks to protobuf
- [x] Loading data
- [ ] Adam optimizer
- [ ] Nestrov + Momentum for GD
- [x] Fix MSE computation in debug mode (not used in actual backpropagation)
- [ ] Somehow persist configurations for Activation, Loss and Optimizer functions in the protobuf messages (???, if we want to do it like it tensorflow, we'd have to do `interface{}` and do type assertions)
- [ ] Convolutional Layers
  - [ ] Flatten layer
- [x] Copy
- [x] Crossover
- [x] Mutate
  - [x] Gaussian Mutator

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
- [x] Unflatten
- [x] NewFromArray - makes a single row
- [x] Map
- [x] Fold
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
- [x] Mean Squared Error
- [ ] Cross Entropy Error (needed for multi class classification!)
- [ ] How to determine how many layers and nodes to use
- [x] One Hot Encoding
- [ ] Convolutional Layers
- [ ] Reinforcment learning
- [x] Genetic Algorithms~
- [x] Neuroevolution~
- [ ] Simulated Annealing
- [ ] Q-Learning
- [ ] Linear vs Logistic Regression

### Questions

These are some (stupid) questions I have that confuse me:

- Is Neuroevolution considered Reinforcement learning?
- How is training done with HUGE datasets when they can't fit on your storage device?
  - Imagine your dataset is a copule of TB big, what do you do?
- Is Q-Learning only done with a single agent (unlike genetic algorithms / neuroevolution)?
  - Is Q-Learning the only method for Reinforcement Learning?
- What's the difference between a Convolutional Neuron and a normal weight matrix?
- Is Deep Learning really just a Neural Network with a lot of layers? (more than 2)
- Why do you need multiple CNN layers? Is it to go to a smaller and smaller version of the image? (when working with images that is) (because of MaxPooling?) Why can't you go directly to the smallest size (512x512 -> 16x16 vs 512x512 -> 256x256 -> 128x128 -> ...)?
- So if images are stored in a 2D array (but with the RGB channels, making it a 3D array with 3 layers), do we use `Conv2D` or `Conv3D`?

### Examples

- [x] XOR Problem
- [x] [Digit Classifier](https://github.com/fr3fou/digit-classifier)
- [x] [Flappy Bird AI](https://github.com/fr3fou/flappy-ai)

### Shoutouts

- [David Josephs](https://github.com/josephsdavid) - was of HUGE help with algebra and other ML-related questions; also helped me spot some nasty bugs!

## References

- <https://www.analyticsvidhya.com/blog/2020/01/fundamentals-deep-learning-activation-functions-when-to-use-them/>
- <https://www.youtube.com/watch?v=XJ7HLz9VYz0&list=PLRqwX-V7Uu6Y7MdSCaIfsxc561QI0U0Tb>
- <https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi>
- <http://matrixmultiplication.xyz/>
- <https://www.khanacademy.org/math/precalculus/x9e81a4f98389efdf:matrices/x9e81a4f98389efdf:properties-of-matrix-addition-and-scalar-multiplication/a/properties-of-matrix-addition>
- <https://www.wikiwand.com/en/Matrix_(mathematics)>
- <https://www.wikiwand.com/en/Activation_function>
- <https://www.wikiwand.com/en/Delta_rule>
- <https://www.jeremyjordan.me/intro-to-neural-networks/>
- <https://www.arxiv-vanity.com/papers/2003.02139/>
- <https://machinelearningmastery.com/gentle-introduction-mini-batch-gradient-descent-configure-batch-size/>
- <http://neuralnetworksanddeeplearning.com/chap2.html>
- <https://arxiv.org/pdf/1802.01528.pdf>
- <https://github.com/stevenmiller888/mind/blob/master/index.js>
- <https://github.com/stevenmiller888/go-mind>
- <https://medium.com/yottabytes/everything-you-need-to-know-about-gradient-descent-applied-to-neural-networks-d70f85e0cc14>
- <https://towardsdatascience.com/deep-learning-which-loss-and-activation-functions-should-i-use-ac02f1c56aa8>
- <https://github.com/milosgajdos/go-neural>
- <https://ml-cheatsheet.readthedocs.io/en/latest/forwardpropagation.html>
- <https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/>
- <https://medium.com/coinmonks/representing-neural-network-with-vectors-and-matrices-c6b0e64db9fb>
- <https://towardsdatascience.com/classifying-cat-pics-with-a-logistic-regression-model-e35dfb9159bb>
- <https://towardsdatascience.com/math-neural-network-from-scratch-in-python-d6da9f29ce65>
- <https://towardsdatascience.com/gradient-descent-from-scratch-e8b75fa986cc>
- <https://rstudio-pubs-static.s3.amazonaws.com/337306_79a7966fad184532ab3ad66b322fe96e.html>
- <https://gombru.github.io/2018/05/23/cross_entropy_loss>
- <https://medium.com/@tomvykruta/memory-aid-for-softmax-and-cross-entropy-loss-5704c66d795d>
- <https://gombru.github.io/2018/05/23/cross_entropy_loss/>
- <https://cs.stackexchange.com/questions/90228/crossover-operator-in-genetic-algorithms-in-neural-networks>
- <https://stackoverflow.com/questions/54625643/where-is-the-gaussian-distribution-function-in-the-pseudocode-below>
- <https://www.wikiwand.com/en/Normal_distribution>
- <https://www.khanacademy.org/math/multivariable-calculus/multivariable-derivatives/partial-derivative-and-gradient-articles/a/introduction-to-partial-derivatives>
- <https://www.khanacademy.org/math/multivariable-calculus/multivariable-derivatives/partial-derivative-and-gradient-articles/a/the-gradient>
