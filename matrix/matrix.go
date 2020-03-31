package matrix

import (
	"fmt"
	"strings"

	"github.com/fr3fou/gone/rand"
)

type Mapper func(val float64, x int, y int) float64
type Folder func(accumulator, val float64, x int, y int) float64

type Matrix struct {
	Rows    int
	Columns int
	Data    [][]float64
}

func New(r, c int, data [][]float64) Matrix {
	if data == nil {
		data = make([][]float64, r)
		for i := range data {
			data[i] = make([]float64, c)
		}
	}

	if len(data) != r {
		panic("matrix: invalid shape of data")
	}

	for _, inputs := range data {
		if len(inputs) != c {
			panic("matrix: invalid shape of data")
		}
	}

	return Matrix{
		Rows:    r,
		Columns: c,
		Data:    data,
	}
}

func NewFromArray(d []float64) Matrix {
	return Map(New(len(d), 1, nil), func(val float64, x, y int) float64 {
		return d[x]
	})
}

func (m *Matrix) Randomize(low, high float64) {
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Columns; j++ {
			m.Data[i][j] = rand.Float(low, high)
		}
	}
}

func (m *Matrix) Zero() {
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Columns; j++ {
			m.Data[i][j] = 0
		}
	}
}

func (m *Matrix) Copy() Matrix {
	return Map(New(m.Rows, m.Columns, nil), func(val float64, x, y int) float64 {
		return m.Data[x][y]
	})
}

func (m Matrix) Map(f Mapper) Matrix {
	return Map(m, f)
}

func (m Matrix) Fold(f Folder, accumulator float64) float64 {
	return Fold(m, f, accumulator)
}

func (m Matrix) Transpose() Matrix {
	return Transpose(m)
}

func (m Matrix) Scale(a float64) Matrix {
	return Scale(m, a)
}

func (m Matrix) AddMatrix(n Matrix) Matrix {
	return AddMatrix(m, n)
}

func (m Matrix) Add(a float64) Matrix {
	return Add(m, a)
}

func (m Matrix) SubtractMatrix(n Matrix) Matrix {
	return SubtractMatrix(m, n)
}

func (m Matrix) Subtract(a float64) Matrix {
	return Subtract(m, a)
}

// HadamardProduct does Hadamard product (entrywise)
func (m Matrix) HadamardProduct(n Matrix) Matrix {
	return HadamardProduct(m, n)
}

// DotProduct does matrix product
func (m Matrix) DotProduct(n Matrix) Matrix {
	return DotProduct(m, n)
}

// Flatten flattens the matrix
func (m Matrix) Flatten() []float64 {
	r := make([]float64, m.Rows*m.Columns)
	i := 0

	for _, line := range m.Data {
		for _, val := range line {
			r[i] = val
			i++
		}
	}

	return r
}

func (m Matrix) String() string {
	b := &strings.Builder{}
	for _, line := range m.Data {
		for _, num := range line {
			fmt.Fprintf(b, " %f ", num)
		}
		b.WriteString("\n")
	}

	return b.String()
}
