package matrix

// Map applies f to every element of the matrix and returns the result
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

// Transpose returns the transposed version of the matrix
func Transpose(m Matrix) Matrix {
	return Map(New(m.Columns, m.Rows, nil),
		func(val float64, x, y int) float64 {
			return m.Data[y][x]
		})
}

// Multiply does scalar multiplication
func Multiply(m Matrix, a float64) Matrix {
	return Map(New(m.Rows, m.Columns, nil), func(val float64, x, y int) float64 {
		return m.Data[x][y] * a
	})
}

// Divide does scalar division
func Divide(m Matrix, a float64) Matrix {
	return Map(New(m.Rows, m.Columns, nil), func(val float64, x, y int) float64 {
		return m.Data[x][y] / a
	})
}

// Sum gives the sum of the elements in the matrix
func Sum(m Matrix) float64 {
	return m.Fold(func(accumulator, val float64, x, y int) float64 {
		return accumulator + val
	}, 0)
}

// AddMatrix adds 2 matrices together
func AddMatrix(m, n Matrix) Matrix {
	if m.Rows != n.Rows || m.Columns != n.Columns {
		panic("matrix: can't add different sized matricies")
	}

	return Map(New(m.Rows, m.Columns, nil), func(val float64, x, y int) float64 {
		return m.Data[x][y] + n.Data[x][y]
	})
}

// Add does scalar addition
func Add(m Matrix, n float64) Matrix {
	return Map(New(m.Rows, m.Columns, nil), func(val float64, x, y int) float64 {
		return m.Data[x][y] + n
	})
}

// SubtractMatrix subtracts 2 matrices
func SubtractMatrix(m, n Matrix) Matrix {
	if m.Rows != n.Rows || m.Columns != n.Columns {
		panic("matrix: can't subtract different sized matricies")
	}

	return Map(New(m.Rows, m.Columns, nil), func(val float64, x, y int) float64 {
		return m.Data[x][y] - n.Data[x][y]
	})
}

// Subtract does scalar subtraction
func Subtract(m Matrix, n float64) Matrix {
	return Map(New(m.Rows, m.Columns, nil), func(val float64, x, y int) float64 {
		return m.Data[x][y] - n
	})
}

// HadamardProduct does Hadamard Product (entrywise)
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
