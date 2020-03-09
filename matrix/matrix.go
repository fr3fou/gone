package matrix

import (
	"fmt"
	"strings"

	"github.com/fr3fou/gone/rand"
)

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
