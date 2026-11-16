// Package onnx reads the subset of the ONNX format Piik's denoiser needs: a
// graph of nodes over float32 initializers. It is a small protobuf reader
// rather than a generated binding, because one model does not justify a
// protobuf dependency and its generated code.
package onnx

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
)

// Model is a parsed ONNX graph.
type Model struct {
	Nodes        []Node
	Initializers map[string]*Tensor
	Inputs       []ValueInfo
	Outputs      []ValueInfo
}

// Node is one operator application.
type Node struct {
	OpType     string
	Inputs     []string
	Outputs    []string
	Attributes map[string]Attribute
}

// Attribute is a node attribute. Only the kinds the graph uses are decoded.
type Attribute struct {
	Ints  []int64
	Int   int64
	Float float32
}

// ValueInfo describes a graph input or output.
type ValueInfo struct {
	Name  string
	Shape []int // -1 marks a symbolic dimension
}

// Tensor is a named float32 initializer.
type Tensor struct {
	Dims []int
	Data []float32
}

// ONNX data types.
const (
	dataTypeFloat  = 1
	dataTypeUint8  = 2
	dataTypeInt8   = 3
	dataTypeUint16 = 4
	dataTypeInt16  = 5
	dataTypeInt32  = 6
	dataTypeInt64  = 7
	dataTypeString = 8
	dataTypeDouble = 11
)

// ErrUnsupported reports a construct this reader does not decode.
var ErrUnsupported = errors.New("onnx: unsupported")

// Parse decodes an ONNX model.
func Parse(payload []byte) (*Model, error) {
	model := &Model{Initializers: map[string]*Tensor{}}
	// ModelProto.graph = 7
	err := walk(payload, func(field int, wire int, value []byte) error {
		if field == 7 && wire == 2 {
			return parseGraph(value, model)
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	return model, nil
}

func parseGraph(payload []byte, model *Model) error {
	return walk(payload, func(field int, wire int, value []byte) error {
		switch {
		case field == 1 && wire == 2: // node
			node, err := parseNode(value)
			if err != nil {
				return err
			}
			model.Nodes = append(model.Nodes, node)
		case field == 5 && wire == 2: // initializer
			tensor, name, err := parseTensor(value)
			if err != nil {
				return err
			}
			model.Initializers[name] = tensor
		case field == 11 && wire == 2: // input
			info, err := parseValueInfo(value)
			if err != nil {
				return err
			}
			model.Inputs = append(model.Inputs, info)
		case field == 12 && wire == 2: // output
			info, err := parseValueInfo(value)
			if err != nil {
				return err
			}
			model.Outputs = append(model.Outputs, info)
		}
		return nil
	})
}

func parseNode(payload []byte) (Node, error) {
	node := Node{Attributes: map[string]Attribute{}}
	err := walk(payload, func(field int, wire int, value []byte) error {
		switch {
		case field == 1 && wire == 2:
			node.Inputs = append(node.Inputs, string(value))
		case field == 2 && wire == 2:
			node.Outputs = append(node.Outputs, string(value))
		case field == 4 && wire == 2:
			node.OpType = string(value)
		case field == 5 && wire == 2:
			attribute, err := parseAttribute(value)
			if err != nil {
				return err
			}
			node.Attributes[attribute.name] = attribute.value
		}
		return nil
	})
	return node, err
}

type namedAttribute struct {
	name  string
	value Attribute
}

func parseAttribute(payload []byte) (namedAttribute, error) {
	var result namedAttribute
	err := walk(payload, func(field int, wire int, value []byte) error {
		switch {
		case field == 1 && wire == 2:
			result.name = string(value)
		case field == 3 && wire == 0:
			parsed, err := varint(value)
			if err != nil {
				return err
			}
			result.value.Int = int64(parsed)
		case field == 8 && wire == 2:
			// Packed repeated int64: a length-delimited run of varints.
			ints, err := packedInts(value)
			if err != nil {
				return err
			}
			result.value.Ints = append(result.value.Ints, ints...)
		case field == 8 && wire == 0:
			// The same field written unpacked, one varint per element.
			parsed, err := varint(value)
			if err != nil {
				return err
			}
			result.value.Ints = append(result.value.Ints, int64(parsed))
		}
		return nil
	})
	return result, err
}

func parseValueInfo(payload []byte) (ValueInfo, error) {
	info := ValueInfo{}
	var typeBytes []byte
	err := walk(payload, func(field int, wire int, value []byte) error {
		switch {
		case field == 1 && wire == 2:
			info.Name = string(value)
		case field == 2 && wire == 2:
			typeBytes = append([]byte(nil), value...)
		}
		return nil
	})
	if err != nil || typeBytes == nil {
		return info, err
	}
	// TypeProto.tensor_type (1) holds elem_type (1) and the shape (2).
	var shapeBytes []byte
	err = walk(typeBytes, func(field int, wire int, value []byte) error {
		if field == 1 && wire == 2 {
			// tensor_type: its shape is field 2.
			_ = walk(value, func(inner int, innerWire int, innerValue []byte) error {
				if inner == 2 && innerWire == 2 {
					shapeBytes = append([]byte(nil), innerValue...)
				}
				return nil
			})
		}
		return nil
	})
	if err != nil || shapeBytes == nil {
		return info, err
	}
	// TensorShapeProto.dim (2) repeats; each holds dim_value (1) or
	// dim_param (2).
	err = walk(shapeBytes, func(field int, wire int, value []byte) error {
		// TensorShapeProto.dim is field 1; each dim message holds dim_value
		// (1) or dim_param (2).
		if field != 1 || wire != 2 {
			return nil
		}
		dim := -1
		if innerErr := walk(value, func(dimField int, dimWire int, dimValue []byte) error {
			if dimField == 1 && dimWire == 0 {
				parsed, parseErr := varint(dimValue)
				if parseErr != nil {
					return parseErr
				}
				dim = int(parsed)
			}
			return nil
		}); innerErr != nil {
			return innerErr
		}
		info.Shape = append(info.Shape, dim)
		return nil
	})
	return info, err
}

func parseTensor(payload []byte) (*Tensor, string, error) {
	tensor := &Tensor{}
	name := ""
	dataType := int64(0)
	var raw []byte
	var floats []float32
	var integers []int64
	err := walk(payload, func(field int, wire int, value []byte) error {
		switch {
		case field == 1 && wire == 0: // dims
			parsed, err := varint(value)
			if err != nil {
				return err
			}
			tensor.Dims = append(tensor.Dims, int(parsed))
		case field == 1 && wire == 2: // packed dims
			ints, err := packedInts(value)
			if err != nil {
				return err
			}
			for _, dim := range ints {
				tensor.Dims = append(tensor.Dims, int(dim))
			}
		case field == 2 && wire == 0: // data_type
			parsed, err := varint(value)
			if err != nil {
				return err
			}
			dataType = int64(parsed)
		case field == 4 && wire == 2: // packed float_data
			for offset := 0; offset+4 <= len(value); offset += 4 {
				floats = append(floats, math.Float32frombits(binary.LittleEndian.Uint32(value[offset:])))
			}
		case field == 4 && wire == 5: // single float_data
			floats = append(floats, math.Float32frombits(binary.LittleEndian.Uint32(value)))
		case field == 5 && wire == 2: // packed int32_data
			values, err := packedInts(value)
			if err != nil {
				return err
			}
			integers = append(integers, values...)
		case field == 7 && wire == 2: // packed int64_data
			values, err := packedInts(value)
			if err != nil {
				return err
			}
			integers = append(integers, values...)
		case field == 7 && wire == 0: // single int64_data
			parsed, err := varint(value)
			if err != nil {
				return err
			}
			integers = append(integers, int64(parsed))
		case field == 8 && wire == 2: // name
			name = string(value)
		case field == 9 && wire == 2: // raw_data
			raw = value
		}
		return nil
	})
	if err != nil {
		return nil, "", err
	}
	size := 1
	for _, dim := range tensor.Dims {
		size *= dim
	}
	switch dataType {
	case dataTypeFloat:
		if len(raw) > 0 {
			if len(raw) != size*4 {
				return nil, "", fmt.Errorf("onnx: tensor %q raw size %d, want %d", name, len(raw), size*4)
			}
			tensor.Data = make([]float32, size)
			for index := 0; index < size; index++ {
				tensor.Data[index] = math.Float32frombits(binary.LittleEndian.Uint32(raw[index*4:]))
			}
			return tensor, name, nil
		}
		tensor.Data = floats
		return tensor, name, nil
	case dataTypeDouble:
		if len(raw) != size*8 {
			return nil, "", fmt.Errorf("onnx: tensor %q double size %d, want %d", name, len(raw), size*8)
		}
		tensor.Data = make([]float32, size)
		for index := 0; index < size; index++ {
			tensor.Data[index] = float32(math.Float64frombits(binary.LittleEndian.Uint64(raw[index*8:])))
		}
		return tensor, name, nil
	case dataTypeInt64, dataTypeInt32:
		// Integer tensors carry shape metadata such as an Unsqueeze axis. They
		// stay float32 so one tensor type serves the graph, but the values are
		// kept: an axis that reads as zero silently reshapes the wrong way.
		width := 8
		if dataType == dataTypeInt32 {
			width = 4
		}
		if len(raw) == size*width {
			tensor.Data = make([]float32, size)
			for index := 0; index < size; index++ {
				offset := index * width
				if width == 8 {
					tensor.Data[index] = float32(int64(binary.LittleEndian.Uint64(raw[offset:])))
				} else {
					tensor.Data[index] = float32(int32(binary.LittleEndian.Uint32(raw[offset:])))
				}
			}
			return tensor, name, nil
		}
		if len(integers) != size {
			return nil, "", fmt.Errorf("onnx: tensor %q has %d integers, want %d", name, len(integers), size)
		}
		tensor.Data = make([]float32, size)
		for index, value := range integers {
			tensor.Data[index] = float32(value)
		}
		return tensor, name, nil
	}
	return nil, "", fmt.Errorf("%w: tensor %q data type %d", ErrUnsupported, name, dataType)
}

// walk iterates protobuf fields, handing each value and its wire type to fn.
func walk(payload []byte, fn func(field int, wire int, value []byte) error) error {
	for len(payload) > 0 {
		key, read := binary.Uvarint(payload)
		if read <= 0 {
			return errors.New("onnx: malformed field key")
		}
		payload = payload[read:]
		field, wire := int(key>>3), int(key&7)
		switch wire {
		case 0: // varint
			value, read := binary.Uvarint(payload)
			if read <= 0 {
				return errors.New("onnx: malformed varint")
			}
			payload = payload[read:]
			if err := fn(field, wire, encodeVarint(value)); err != nil {
				return err
			}
		case 1: // fixed64
			if len(payload) < 8 {
				return errors.New("onnx: truncated fixed64")
			}
			if err := fn(field, wire, payload[:8]); err != nil {
				return err
			}
			payload = payload[8:]
		case 2: // length-delimited
			length, read := binary.Uvarint(payload)
			if read <= 0 || uint64(len(payload)-read) < length {
				return errors.New("onnx: truncated field")
			}
			payload = payload[read:]
			if err := fn(field, wire, payload[:length]); err != nil {
				return err
			}
			payload = payload[length:]
		case 5: // fixed32
			if len(payload) < 4 {
				return errors.New("onnx: truncated fixed32")
			}
			if err := fn(field, wire, payload[:4]); err != nil {
				return err
			}
			payload = payload[4:]
		default:
			return fmt.Errorf("%w: wire type %d", ErrUnsupported, wire)
		}
	}
	return nil
}

func varint(payload []byte) (uint64, error) {
	value, read := binary.Uvarint(payload)
	if read <= 0 {
		return 0, errors.New("onnx: malformed varint")
	}
	return value, nil
}

func packedInts(payload []byte) ([]int64, error) {
	var ints []int64
	for len(payload) > 0 {
		value, read := binary.Uvarint(payload)
		if read <= 0 {
			return nil, errors.New("onnx: malformed packed varint")
		}
		ints = append(ints, int64(value))
		payload = payload[read:]
	}
	return ints, nil
}

func encodeVarint(value uint64) []byte {
	var buffer [binary.MaxVarintLen64]byte
	length := binary.PutUvarint(buffer[:], value)
	return buffer[:length]
}
