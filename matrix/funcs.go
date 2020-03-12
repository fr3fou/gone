package matrix

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
	return Map(New(m.Columns, m.Rows, nil),
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
	if m.Rows != n.Rows || m.Columns != n.Columns {
		panic("matrix: can't add different sized matricies")
	}

	return Map(m, func(val float64, x, y int) float64 {
		return val + n.Data[x][y]
	})
}

func Add(m Matrix, n float64) Matrix {
	return Map(m, func(val float64, x, y int) float64 {
		return val + n
	})
}

func SubtractMatrix(m, n Matrix) Matrix {
	if m.Rows != n.Rows || m.Columns != n.Columns {
		panic("matrix: can't subtract different sized matricies")
	}

	return Map(m, func(val float64, x, y int) float64 {
		return val - n.Data[x][y]
	})
}

func Subtract(m Matrix, n float64) Matrix {
	return Map(m, func(val float64, x, y int) float64 {
		return val - n
	})
}

// Multiply does Hadamard Product
func HadamardProduct(m Matrix, n Matrix) Matrix {
	if m.Columns != n.Columns || m.Rows != n.Rows {
		panic("matrix: matricies must have the same shape")
	}

	return Map(m, func(val float64, x, y int) float64 {
		return val * n.Data[x][y]
	})
}

// DotProduct does matrix product
func DotProduct(m, n Matrix) Matrix {
	if m.Columns != n.Rows {
		panic("matrix: rows must match with columns of matricies")
	}

	return Map(New(m.Rows, n.Columns, nil), func(_ float64, x, y int) float64 {
		sum := 0.0

		for i := 0; i < n.Rows; i++ {
			sum += m.Data[x][i] * n.Data[i][y]
		}

		return sum
	})
}
