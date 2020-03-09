package matrix

import "github.com/fr3fou/gone/rand"

type Matrix struct {
	Rows    int
	Columns int
	Data    [][]float64
}

func New(r, c int, data [][]float64) Matrix {
	if data == nil {
		data = [][]float64{}
	}

	return Matrix{
		Rows:    r,
		Columns: c,
		Data:    data,
	}
}

func (m *Matrix) Randomize() {
	for i := range m.Data {
		for j := range m.Data[i] {
			m.Data[i][j] = rand.Float(-50, 50)
		}
	}
}
