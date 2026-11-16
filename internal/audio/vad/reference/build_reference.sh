#!/usr/bin/env bash
# Builds WebRTC's rnn_vad_tool against the prebuilt libraries from
# https://github.com/wireapp/prebuilt-webrtc-binaries and runs it on a WAV.
#
# This has to run on Linux: the prebuilt archives are linux-only, and WebRTC
# cannot be built from a Windows checkout. Run it in a container or on a Linux
# host, then copy the .bin output back into the fixtures directory.
#
# Usage: ./build_reference.sh <input.wav> <output-prefix>
#   writes <output-prefix>_features.bin and <output-prefix>_vad.bin

set -euo pipefail

WEBRTC_VERSION="${WEBRTC_VERSION:-20260605.102}"
WORK="${WORK:-$PWD/webrtc-build}"
INPUT="${1:?usage: build_reference.sh <input.wav> <output-prefix>}"
OUTPUT="${2:?usage: build_reference.sh <input.wav> <output-prefix>}"

mkdir -p "$WORK"
cd "$WORK"

if [ ! -d webrtc ]; then
  echo "downloading WebRTC $WEBRTC_VERSION"
  curl -fsSL -o linux.zip \
    "https://github.com/wireapp/prebuilt-webrtc-binaries/releases/download/$WEBRTC_VERSION/webrtc_${WEBRTC_VERSION}_linux.zip"
  curl -fsSL -o headers.zip \
    "https://github.com/wireapp/prebuilt-webrtc-binaries/releases/download/$WEBRTC_VERSION/webrtc_${WEBRTC_VERSION}_headers.zip"
  mkdir -p webrtc
  unzip -q -o linux.zip -d webrtc
  unzip -q -o headers.zip -d webrtc
fi

# The harness mirrors rnn_vad_tool.cc: 10 ms frames, resampled to 24 kHz, then
# features and probabilities written as raw float32.
cat > harness.cc <<'HARNESS'
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "common_audio/wav_file.h"
#include "modules/audio_processing/agc2/rnn_vad/features_extraction.h"
#include "modules/audio_processing/agc2/rnn_vad/rnn.h"

using namespace webrtc;
using namespace webrtc::rnn_vad;

int main(int argc, char** argv) {
  if (argc < 4) {
    fprintf(stderr, "usage: %s <in.wav> <features.bin> <vad.bin>\n", argv[0]);
    return 1;
  }
  WavReader reader(argv[1]);
  if (reader.num_channels() != 1) {
    fprintf(stderr, "mono only\n");
    return 1;
  }
  FILE* features = fopen(argv[2], "wb");
  FILE* vad = fopen(argv[3], "wb");
  if (!features || !vad) {
    fprintf(stderr, "cannot open output\n");
    return 1;
  }

  const int frame_size = reader.sample_rate() / 100;
  std::vector<float> samples(frame_size);
  std::array<float, kFrameSize10ms24kHz> samples_24k;
  FeaturesExtractor extractor(GetAvailableCpuFeatures());
  RnnVad network(GetAvailableCpuFeatures());
  std::array<float, kFeatureVectorSize> feature_vector;

  while (reader.ReadSamples(frame_size, samples.data()) ==
         static_cast<size_t>(frame_size)) {
    // The prebuilt build has no resampler in this harness; pass the frame
    // straight through when the input is already 24 kHz.
    if (reader.sample_rate() != kSampleRate24kHz) {
      fprintf(stderr, "input must be 24 kHz (got %d)\n", reader.sample_rate());
      return 1;
    }
    for (int i = 0; i < kFrameSize10ms24kHz; ++i) samples_24k[i] = samples[i];
    const bool silence =
        extractor.CheckSilenceComputeFeatures(samples_24k, feature_vector);
    const float probability =
        network.ComputeVadProbability(feature_vector, silence);
    fwrite(feature_vector.data(), sizeof(float), kFeatureVectorSize, features);
    fwrite(&probability, sizeof(float), 1, vad);
  }
  fclose(features);
  fclose(vad);
  return 0;
}
HARNESS

INCLUDES="-Iwebrtc/include -Iwebrtc"
LIBS="$(find webrtc -name '*.a' | tr '\n' ' ')"

g++ -std=c++17 -O2 -o harness harness.cc $INCLUDES \
  -Wl,--start-group $LIBS -Wl,--end-group -lpthread -ldl -lm

./harness "$INPUT" "${OUTPUT}_features.bin" "${OUTPUT}_vad.bin"
echo "wrote ${OUTPUT}_features.bin and ${OUTPUT}_vad.bin"
