// Copyright (c) 2008-2011 Octasic Inc.
//               2012-2017 Jean-Marc Valin
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// - Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// - Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES ARE DISCLAIMED. See the rnnoise COPYING
// file for the full text.
//
// This package is a Go port of the voice activity detector that WebRTC vendors
// from rnnoise for AGC2 (modules/audio_processing/agc2/rnn_vad). The network is
// a dense layer, a GRU and an output layer over 42 features; the weights are
// the ones rnnoise ships in src/rnn_data.c.

package vad

import (
	_ "embed"
	"encoding/binary"
	"math"
)

// Network dimensions, from rnnoise's rnn_data.c.
const (
	// FeatureSize is the number of input features per frame.
	FeatureSize = 42
	// DenseSize is the width of the input dense layer.
	DenseSize = 24
	// GRUSize is the width of the recurrent layer. rnnoise's VAD GRU is 24
	// wide; the 72-wide table belongs to the denoiser branch, which this port
	// does not use.
	GRUSize = 24
)

// weightsScale converts the int8 weights to floats, matching rnnoise's
// WEIGHTS_SCALE.
const weightsScale = 1.0 / 256

//go:embed weights.bin
var weightData []byte

// Network is the VAD's recurrent network. It keeps the GRU state between
// frames, so one network belongs to one stream.
type Network struct {
	inputWeights []float32 // DenseSize x FeatureSize
	inputBias    []float32 // DenseSize
	gruWeights   []float32 // 3*GRUSize x DenseSize
	gruRecurrent []float32 // 3*GRUSize x GRUSize
	gruBias      []float32 // 3*GRUSize
	outWeights   []float32 // DenseSize
	outBias      float32
	state        []float32 // GRUSize
}

// NewNetwork loads the embedded weights and clears the recurrent state.
func NewNetwork() *Network {
	network := &Network{}
	offset := 0
	read := func(count int) []float32 {
		values := make([]float32, count)
		for index := range values {
			values[index] = float32(int16(binary.LittleEndian.Uint16(weightData[offset+index*2:]))) * weightsScale
		}
		offset += count * 2
		return values
	}
	network.inputWeights = read(DenseSize * FeatureSize)
	network.inputBias = read(DenseSize)
	// Each table holds 3*GRUSize rows of DenseSize (or GRUSize) columns, which
	// is 1728 values each for the 24-wide VAD GRU.
	network.gruWeights = read(3 * GRUSize * DenseSize)
	network.gruRecurrent = read(3 * GRUSize * GRUSize)
	network.gruBias = read(3 * GRUSize)
	network.outWeights = read(DenseSize)
	network.outBias = read(1)[0]
	network.state = make([]float32, GRUSize)
	return network
}

// Reset clears the recurrent state, so one utterance does not colour the next.
func (network *Network) Reset() {
	for index := range network.state {
		network.state[index] = 0
	}
}

// tansig approximates tanh the way rnnoise does: a 201-entry table plus a
// correction term. Matching the approximation matters because AGC2's
// thresholds were tuned against it.
func tansig(x float32) float32 {
	if !(x < 8) {
		return 1
	}
	if !(x > -8) {
		return -1
	}
	sign := float32(1)
	if x < 0 {
		x = -x
		sign = -1
	}
	index := int(math.Floor(0.5 + 25*float64(x)))
	if index > 200 {
		index = 200
	}
	x -= 0.04 * float32(index)
	y := tansigTable[index]
	dy := 1 - y*y
	y = y + x*dy*(1-y*x)
	return sign * y
}

func sigmoid(x float32) float32 { return 0.5 + 0.5*tansig(0.5*x) }

// Probability runs one frame of features through the network and returns the
// speech probability.
func (network *Network) Probability(features []float32) float32 {
	if len(features) != FeatureSize {
		return 0
	}
	// Dense layer with tanh, matching rnnoise's input_dense.
	dense := make([]float32, DenseSize)
	for unit := 0; unit < DenseSize; unit++ {
		sum := network.inputBias[unit]
		for input := 0; input < FeatureSize; input++ {
			sum += network.inputWeights[input*DenseSize+unit] * features[input]
		}
		dense[unit] = tansig(sum)
	}
	network.gru(dense)
	// Output layer with sigmoid, matching rnnoise's vad_output.
	var sum float32
	for unit := 0; unit < DenseSize; unit++ {
		sum += network.outWeights[unit] * network.state[unit]
	}
	return sigmoid(sum + network.outBias)
}

// gru advances the recurrent layer. rnnoise stores the three gates interleaved
// with a stride of 3*GRUSize, in the order update, reset, output.
func (network *Network) gru(input []float32) {
	const stride = GRUSize * 3
	update := make([]float32, GRUSize)
	reset := make([]float32, GRUSize)
	output := make([]float32, GRUSize)
	for unit := 0; unit < GRUSize; unit++ {
		sum := network.gruBias[unit]
		for index := 0; index < DenseSize; index++ {
			sum += network.gruWeights[index*stride+unit] * input[index]
		}
		for index := 0; index < GRUSize; index++ {
			sum += network.gruRecurrent[index*stride+unit] * network.state[index]
		}
		update[unit] = sigmoid(sum)
	}
	for unit := 0; unit < GRUSize; unit++ {
		sum := network.gruBias[GRUSize+unit]
		for index := 0; index < DenseSize; index++ {
			sum += network.gruWeights[GRUSize+index*stride+unit] * input[index]
		}
		for index := 0; index < GRUSize; index++ {
			sum += network.gruRecurrent[GRUSize+index*stride+unit] * network.state[index]
		}
		reset[unit] = sigmoid(sum)
	}
	for unit := 0; unit < GRUSize; unit++ {
		sum := network.gruBias[2*GRUSize+unit]
		for index := 0; index < DenseSize; index++ {
			sum += network.gruWeights[2*GRUSize+index*stride+unit] * input[index]
		}
		for index := 0; index < GRUSize; index++ {
			sum += network.gruRecurrent[2*GRUSize+index*stride+unit] * network.state[index] * reset[index]
		}
		// rnnoise's vad_gru uses RELU, not tanh.
		if sum < 0 {
			sum = 0
		}
		output[unit] = update[unit]*network.state[unit] + (1-update[unit])*sum
	}
	copy(network.state, output)
}
