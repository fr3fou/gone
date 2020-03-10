package matrix

import (
	"fmt"
	"strings"

	"github.com/fr3fou/gone/rand"
)

type Mapper func(val float64, x int, y int) float64

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

	return Matrix{
		Rows:    r,
		Columns: c,
		Data:    data,
	}
}

func (m *Matrix) Randomize(low, high float64) {
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Columns; j++ {
			m.Data[i][j] = rand.Float(low, high)
		}
	}
}

func Map(m Matrix, f Mapper) Matrix {
	n := New(m.Rows, m.Columns, m.Data)

	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Columns; j++ {
			val := n.Data[i][j]
			n.Data[i][j] = f(val, i, j)
		}
	}

	return n
}

func Transpose(m Matrix) Matrix {
	return Map(New(m.Rows, m.Columns, nil),
		func(val float64, x, y int) float64 {
			return m.Data[y][x]
		})
}

func Scale(m Matrix, a float64) Matrix {
	return Map(m, func(val float64, x, y int) float64 {
		return val * a
	})
}

func AddMatrix(m, n Matrix) Matrix {
	if m.Rows != n.Rows || m.Columns != n.Rows {
		panic("matrix: can't add different sized matricies")
	}

	result := New(m.Rows, m.Columns, m.Data)

	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Columns; j++ {
			result.Data[i][j] += n.Data[i][j]
		}
	}

	return result
}

func Add(m Matrix, n float64) Matrix {
	result := New(m.Rows, m.Columns, m.Data)

	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Columns; j++ {
			result.Data[i][j] += n
		}
	}

	return result
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
