package onnx

import _ "embed"

// ModelBytes is the production DFSMN-ANS graph. The model is committed in
// testdata and embedded here so a shipped binary can load the denoiser without
// a runtime model path.
//
//go:embed testdata/dfsmn_ans.onnx
var ModelBytes []byte
