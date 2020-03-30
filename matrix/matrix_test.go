package matrix

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestTranspose(t *testing.T) {
	m := New(3, 3, [][]float64{
		{1, 2, 3},
		{4, 5, 6},
		{7, 8, 9},
	})

	trans := New(3, 3, [][]float64{
		{1, 4, 7},
		{2, 5, 8},
		{3, 6, 9},
	})

	assert.Equal(t, trans, Transpose(m))
}

func TestScale(t *testing.T) {
	m := New(3, 3, [][]float64{
		{1, 2, 3},
		{4, 5, 6},
		{7, 8, 9},
	})

	expected := New(3, 3, [][]float64{
		{1 * 0.5, 2 * 0.5, 3 * 0.5},
		{4 * 0.5, 5 * 0.5, 6 * 0.5},
		{7 * 0.5, 8 * 0.5, 9 * 0.5},
	})

	assert.Equal(t, expected, Scale(m, 0.5))
}

func TestAddMatrix(t *testing.T) {
	m := New(3, 3, [][]float64{
		{1, 2, 3},
		{4, 5, 6},
		{7, 8, 9},
	})

	n := New(3, 3, [][]float64{
		{9, 8, 7},
		{6, 5, 4},
		{3, 2, 1},
	})

	expected := New(3, 3, [][]float64{
		{10, 10, 10},
		{10, 10, 10},
		{10, 10, 10},
	})

	assert.Equal(t, expected, AddMatrix(m, n))
}

func TestAdd(t *testing.T) {
	m := New(3, 3, [][]float64{
		{1, 2, 3},
		{4, 5, 6},
		{7, 8, 9},
	})

	expected := New(3, 3, [][]float64{
		{6, 7, 8},
		{9, 10, 11},
		{12, 13, 14},
	})

	assert.Equal(t, expected, Add(m, 5))
}

func TestSubtractMatrix(t *testing.T) {
	m := New(3, 3, [][]float64{
		{1, 2, 3},
		{4, 5, 6},
		{7, 8, 9},
	})

	n := New(3, 3, [][]float64{
		{1, 2, 3},
		{4, 5, 6},
		{7, 8, 9},
	})

	expected := New(3, 3, nil)

	assert.Equal(t, expected, SubtractMatrix(m, n))
}

func TestSubtract(t *testing.T) {
	m := New(3, 3, [][]float64{
		{10, 10, 10},
		{10, 10, 10},
		{10, 10, 10},
	})

	expected := New(3, 3, nil)

	assert.Equal(t, expected, Subtract(m, 10))
}

func TestDotProduct(t *testing.T) {
	m := New(3, 3, [][]float64{
		{1, 2, 3},
		{4, 5, 6},
		{7, 8, 9},
	})

	n := New(3, 3, [][]float64{
		{9, 8, 7},
		{6, 5, 4},
		{3, 2, 1},
	})

	expected := New(3, 3, [][]float64{
		{30, 24, 18},
		{84, 69, 54},
		{138, 114, 90},
	})

	assert.Equal(t, expected, DotProduct(m, n))
}

func TestMultiply(t *testing.T) {
	m := New(2, 1, [][]float64{
		{1},
		{2},
	})

	n := New(2, 1, [][]float64{
		{3},
		{4},
	})

	expected := New(2, 1, [][]float64{
		{3},
		{8},
	})

	assert.Equal(t, expected, HadamardProduct(m, n))
}

func TestFlatten(t *testing.T) {
	{
		m := New(3, 3, [][]float64{
			{1, 2, 3},
			{4, 5, 6},
			{7, 8, 9},
		})

		expected := []float64{1, 2, 3, 4, 5, 6, 7, 8, 9}

		assert.Equal(t, expected, m.Flatten())
	}

	{
		m := New(3, 1, [][]float64{
			{1},
			{4},
			{7},
		})

		expected := []float64{1, 4, 7}

		assert.Equal(t, expected, m.Flatten())
	}

	{
		m := New(1, 3, [][]float64{
			{1, 2, 3},
		})

		expected := []float64{1, 2, 3}

		assert.Equal(t, expected, m.Flatten())
	}
}

func TestNewFromArray(t *testing.T) {
	m := NewFromArray([]float64{1, 2, 3, 4, 5, 6, 7, 8, 9})

	expected := New(9, 1, [][]float64{
		{1},
		{2},
		{3},
		{4},
		{5},
		{6},
		{7},
		{8},
		{9},
	})

	assert.Equal(t, expected, m)
}
