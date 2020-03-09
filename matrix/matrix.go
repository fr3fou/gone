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

func (m *Matrix) Randomize() {
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Columns; j++ {
			m.Data[i][j] = rand.Float(-50, 50)
		}
	}
}

func (m Matrix) String() string {
	b := &strings.Builder{}
	for i, line := range m.Data {
		if i%m.Rows == 0 {
			fmt.Fprintln(b, "+---------+---------+---------+")
		}
		for j, num := range line {
			if j%m.Columns == 0 {
				fmt.Fprint(b, "|")

			}
			if num == 0 {
				fmt.Fprint(b, " . ")

			} else {
				fmt.Fprintf(b, " %f ", num)

			}
			if j == 8 {
				fmt.Fprint(b, "|")

			}

		}
		if i == m.Columns-1 {
			fmt.Fprint(b, "\n+---------+---------+---------+")

		}
		fmt.Fprintln(b)
	}

	return b.String()
}
