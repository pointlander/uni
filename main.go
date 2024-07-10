// Copyright 2024 The Uni Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/pointlander/matrix"
)

const (
	// S is the scaling factor for the softmax
	S = 1.0 - 1e-300
)

func softmax(values []float32) {
	max := float32(0.0)
	for _, v := range values {
		if v > max {
			max = v
		}
	}
	s := max * S
	sum := float32(0.0)
	for j, value := range values {
		values[j] = float32(math.Exp(float64(value - s)))
		sum += values[j]
	}
	for j, value := range values {
		values[j] = value / sum
	}
}

func dot32(a, b []float32) float32 {
	sum := 0.0
	for i, v := range a {
		sum += float64(v) * float64(b[i])
	}
	return float32(sum)
}

// SelfAttention computes the self attention of Q, K, V
func SelfAttention(Q, K, V matrix.Matrix) matrix.Matrix {
	o := matrix.Matrix{
		Cols: V.Cols,
		Rows: K.Rows,
		Data: make([]float32, 0, V.Rows*K.Rows),
	}
	outputs, values := make([]float32, V.Cols), make([]float32, Q.Rows)
	V = V.T()
	for i := 0; i < K.Rows; i++ {
		K := K.Data[i*K.Cols : (i+1)*K.Cols]
		for j := 0; j < Q.Rows; j++ {
			Q := Q.Data[j*Q.Cols : (j+1)*Q.Cols]
			values[j] = dot32(K, Q)
		}
		softmax(values)
		for j := 0; j < V.Rows; j++ {
			V := V.Data[j*V.Cols : (j+1)*V.Cols]
			outputs[j] = dot32(values, V)
		}
		o.Data = append(o.Data, outputs...)
	}
	return o
}

func main() {
	rng := rand.New(rand.NewSource(1))
	input := matrix.NewMatrix(3, 1)
	query := matrix.NewMatrix(3, 3)
	key := matrix.NewMatrix(3, 3)
	value := matrix.NewMatrix(3, 3)
	for i := 0; i < 3*3; i++ {
		query.Data = append(query.Data, float32(rng.NormFloat64()))
		key.Data = append(key.Data, float32(rng.NormFloat64()))
		value.Data = append(value.Data, float32(rng.NormFloat64()))
	}
	for i := 0; i < input.Cols; i++ {
		input.Data = append(input.Data, float32(rng.NormFloat64()))
	}
	for i := 0; i < 8; i++ {
		output := SelfAttention(query.MulT(input), key.MulT(input), value.MulT(input))
		newInput := matrix.NewMatrix(input.Cols, input.Rows+output.Rows)
		newInput.Data = append(newInput.Data, input.Data...)
		newInput.Data = append(newInput.Data, output.Data...)
		input = newInput
	}
	for i := 0; i < input.Rows; i++ {
		for j := 0; j < input.Cols; j++ {
			fmt.Printf("%f ", input.Data[i*input.Cols+j])
		}
		fmt.Println()
	}
}
