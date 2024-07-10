// Copyright 2024 The Uni Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"math/rand"

	"github.com/pointlander/matrix"
)

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
		output := matrix.SelfAttention(query.MulT(input), key.MulT(input), value.MulT(input))
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
