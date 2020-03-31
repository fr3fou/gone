package matrix

func Map(m Matrix, f Mapper) Matrix {
	n := New(m.Rows, m.Columns, nil)

	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Columns; j++ {
			val := m.Data[i][j]
			n.Data[i][j] = f(val, i, j)
		}
	}

	return n
}

// Fold accumulates the values in a matrix according to a Folder function
func Fold(m Matrix, f Folder, accumulator float64) float64 {
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Columns; j++ {
			accumulator = f(accumulator, m.Data[i][j], i, j)
		}
	}

	return accumulator
}

func Transpose(m Matrix) Matrix {
	return Map(New(m.Columns, m.Rows, nil),
		func(val float64, x, y int) float64 {
			return m.Data[y][x]
		})
}

func Scale(m Matrix, a float64) Matrix {
	return Map(New(m.Rows, m.Columns, nil), func(val float64, x, y int) float64 {
		return m.Data[x][y] * a
	})
}

func AddMatrix(m, n Matrix) Matrix {
	if m.Rows != n.Rows || m.Columns != n.Columns {
		panic("matrix: can't add different sized matricies")
	}

	return Map(New(m.Rows, m.Columns, nil), func(val float64, x, y int) float64 {
		return m.Data[x][y] + n.Data[x][y]
	})
}

func Add(m Matrix, n float64) Matrix {
	return Map(New(m.Rows, m.Columns, nil), func(val float64, x, y int) float64 {
		return m.Data[x][y] + n
	})
}

func SubtractMatrix(m, n Matrix) Matrix {
	if m.Rows != n.Rows || m.Columns != n.Columns {
		panic("matrix: can't subtract different sized matricies")
	}

	return Map(New(m.Rows, m.Columns, nil), func(val float64, x, y int) float64 {
		return m.Data[x][y] - n.Data[x][y]
	})
}

func Subtract(m Matrix, n float64) Matrix {
	return Map(New(m.Rows, m.Columns, nil), func(val float64, x, y int) float64 {
		return m.Data[x][y] - n
	})
}

// HadamardProduct does Hadamard Product
func HadamardProduct(m Matrix, n Matrix) Matrix {
	if m.Columns != n.Columns || m.Rows != n.Rows {
		panic("matrix: matricies must have the same shape")
	}

	return Map(New(m.Rows, m.Columns, nil), func(val float64, x, y int) float64 {
		return m.Data[x][y] * n.Data[x][y]
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
