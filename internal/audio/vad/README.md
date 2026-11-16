# Voice activity detection

A Go port of the voice activity detector WebRTC vendors from rnnoise for AGC2
(`modules/audio_processing/agc2/rnn_vad`). It exists so Piik's voice pipeline can
gate its automatic gain control on speech the way WebRTC does, without cgo and
without vendoring WebRTC.

## What is here

| File | Contents | Verified against WebRTC? |
| --- | --- | --- |
| `vad.go` | the network: dense (42→24, tanh) → GRU (24, relu) → dense (24→1, sigmoid) | **yes**, 1e-06 |
| `weights.bin` | 4,585 int8 weights from rnnoise `src/rnn_data.c`, embedded | **byte-identical** |
| `tansig.go` | rnnoise's 201-entry approximated tanh table | matches rnnoise |
| `extractor.go` | the full front end: pitch buffer, LP residual, pitch, spectral features | **yes**, 100% of features within 0.1 |
| `pitch.go` | LP residual, 12 kHz search, 24 kHz refinement, extended search | **yes**, period exact |
| `spectral.go` | Opus bands, cepstrum, derivatives, cross-correlation, variability | **yes** |
| `dft.go` | exact 480-point DFT | matches PFFFT |
| `math32.go` | float32 math wrappers | — |

## Verification against real WebRTC

WebRTC's AGC2 VAD was compiled from source in a Linux container and run on
`poem_24k.wav`, producing `webrtc_poem_features.f32` and `webrtc_poem_vad.f32`
in the fixtures directory. Three tests compare against it:

| Test | Result |
| --- | --- |
| `TestNetworkMatchesWebRTC` | max probability error **1e-06** over 1,268 frames |
| `TestFeaturesMatchWebRTC` | **100.0%** of feature values within 0.1; max 0.25 |
| `TestEndToEndMatchesWebRTC` | max probability error **0.0051** end to end |

The residual 0.25 is float32 accumulation order in the cross-correlation bands;
it is below the precision the network can distinguish.

## Why the port needed float32

WebRTC computes the whole front end in `float`. A `float64` port produced
different LPC coefficients in the last digits, which flipped near-ties in the
pitch search and produced octave errors. Converting `pitch.go`, `spectral.go`,
`dft.go` and `extractor.go` to `float32` was required for an exact match.

## Reference build

`D:\Document\piik-work\refbuild` holds the WebRTC source tree, the harnesses
that dump intermediates, and the built reference. It is not part of the module;
regenerate the fixtures there if the reference needs rebuilding.
