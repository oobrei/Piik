"""Reference implementation of WebRTC's AGC2 VAD feature extraction.

Translated from modules/audio_processing/agc2/rnn_vad/ in the WebRTC tree:
  spectral_features.cc, spectral_features_internal.cc, features_extraction.cc

This is an oracle for the Go port in internal/audio/vad, not a production
implementation. It is slow, single-threaded and allocates freely.

Copyright (c) 2018 The WebRTC project authors. All Rights Reserved.
Use of this source code is governed by a BSD-style license that can be found in
the LICENSE file in the root of the source tree. An additional intellectual
property rights grant can be found in the PATENTS file. All contributing
project authors may be found in the AUTHORS file in the source tree.
"""

import math

import numpy as np

# common.h
FRAME_10MS_24K = 240
FRAME_20MS_24K = 480
NUM_BANDS = 22
NUM_LOWER_BANDS = 6
CEPSTRAL_HISTORY = 8
FEATURE_VECTOR_SIZE = 42
OPUS_BANDS_24K = 20
SILENCE_THRESHOLD = 0.04

# spectral_features_internal.h
OPUS_SCALE_NUM_BINS = [4, 4, 4, 4, 4, 4, 4, 4, 8, 8, 8, 8, 16, 16, 16, 24, 24, 32, 48]

# spectral_features_internal.cc, kOpusBandWeights24kHz20ms. The triangular
# filter weights, one per FFT coefficient, grouped by band.
OPUS_BAND_WEIGHTS = []


def _build_band_weights():
    """Rebuilds the hard-coded weight table from its structure.

    The table is the triangular filter for each band: within a band of n bins
    the weight for bin j is j/n, so band 0 (4 bins) is 0, 0.25, 0.5, 0.75.
    """
    weights = []
    for size in OPUS_SCALE_NUM_BINS:
        for j in range(size):
            weights.append(j / size)
    return weights


OPUS_BAND_WEIGHTS = _build_band_weights()


def _vorbis_half_window():
    """ComputeScaledHalfVorbisWindow(1/480)."""
    half = FRAME_20MS_24K // 2
    scaling = 1.0 / FRAME_20MS_24K
    window = np.empty(half, dtype=np.float64)
    for i in range(half):
        s = math.sin(0.5 * math.pi * (i + 0.5) / half)
        window[i] = scaling * math.sin(0.5 * math.pi * s * s)
    return window


HALF_WINDOW = _vorbis_half_window()


def _dct_table():
    """ComputeDctTable()."""
    table = np.empty((NUM_BANDS, NUM_BANDS), dtype=np.float64)
    k = math.sqrt(0.5)
    for i in range(NUM_BANDS):
        for j in range(NUM_BANDS):
            table[i, j] = math.cos((i + 0.5) * j * math.pi / NUM_BANDS)
        table[i, 0] *= k
    return table


DCT_TABLE = _dct_table()
DCT_SCALING = 0.301511345


def windowed_fft(frame):
    """ComputeWindowedForwardFft, returning the interleaved spectrum.

    The window is mirrored across the frame and the Nyquist coefficient is
    zeroed, matching WebRTC.
    """
    half = FRAME_20MS_24K // 2
    windowed = np.empty(FRAME_20MS_24K, dtype=np.float64)
    for i in range(half):
        j = FRAME_20MS_24K - 1 - i
        windowed[i] = frame[i] * HALF_WINDOW[i]
        windowed[j] = frame[j] * HALF_WINDOW[i]
    spectrum = np.fft.rfft(windowed)
    # WebRTC stores real/imaginary pairs for the 240 non-Nyquist coefficients;
    # the Nyquist coefficient is dropped rather than stored.
    interleaved = np.empty(FRAME_20MS_24K, dtype=np.float64)
    interleaved[0::2] = spectrum.real[: FRAME_20MS_24K // 2]
    interleaved[1::2] = spectrum.imag[: FRAME_20MS_24K // 2]
    interleaved[1] = 0.0  # Nyquist coefficient
    return interleaved


def cross_correlation(x, y):
    """SpectralCorrelator::ComputeCrossCorrelation."""
    out = np.zeros(OPUS_BANDS_24K, dtype=np.float64)
    k = 0
    for i in range(OPUS_BANDS_24K - 1):
        for _ in range(OPUS_SCALE_NUM_BINS[i]):
            v = x[2 * k] * y[2 * k] + x[2 * k + 1] * y[2 * k + 1]
            tmp = OPUS_BAND_WEIGHTS[k] * v
            out[i] += v - tmp
            out[i + 1] += tmp
            k += 1
    out[0] *= 2.0
    return out


def smoothed_log_magnitude(bands_energy):
    """ComputeSmoothedLogMagnitudeSpectrum."""
    log_bands = np.empty(NUM_BANDS, dtype=np.float64)
    log_max = -2.0
    follow = -2.0

    def smooth(x):
        nonlocal log_max, follow
        x = max(log_max - 7.0, follow - 1.5, x)
        log_max = max(log_max, x)
        follow = max(follow - 1.5, x)
        return x

    for i in range(len(bands_energy)):
        log_bands[i] = smooth(math.log10(1e-2 + bands_energy[i]))
    for i in range(len(bands_energy), NUM_BANDS):
        log_bands[i] = smooth(-2.0)
    return log_bands


def dct(values):
    """ComputeDct."""
    out = np.zeros(NUM_BANDS, dtype=np.float64)
    for i in range(len(values)):
        total = 0.0
        for j in range(len(values)):
            total += values[j] * DCT_TABLE[j, i]
        out[i] = total * DCT_SCALING
    return out


class SpectralFeatures:
    """The subset of SpectralFeaturesExtractor this oracle needs."""

    def __init__(self):
        self.cepstral_ring = [np.zeros(NUM_BANDS) for _ in range(CEPSTRAL_HISTORY)]
        self.cepstral_diffs = np.zeros((CEPSTRAL_HISTORY, CEPSTRAL_HISTORY))

    def reset(self):
        self.cepstral_ring = [np.zeros(NUM_BANDS) for _ in range(CEPSTRAL_HISTORY)]
        self.cepstral_diffs = np.zeros((CEPSTRAL_HISTORY, CEPSTRAL_HISTORY))

    def compute(self, reference, lagged):
        """Returns (is_silence, features) where features is 42 floats."""
        ref_fft = windowed_fft(reference)
        ref_energy = cross_correlation(ref_fft, ref_fft)
        if ref_energy.sum() < SILENCE_THRESHOLD:
            return True, np.zeros(FEATURE_VECTOR_SIZE)

        lag_fft = windowed_fft(lagged)
        lag_energy = cross_correlation(lag_fft, lag_fft)

        log_bands = smoothed_log_magnitude(ref_energy)
        cepstrum = dct(log_bands)
        cepstrum[0] -= 12.0
        cepstrum[1] -= 4.0

        self.cepstral_ring.insert(0, cepstrum)
        self.cepstral_ring.pop()
        # UpdateCepstralDifferenceStats
        distances = np.zeros(CEPSTRAL_HISTORY - 1)
        for i in range(CEPSTRAL_HISTORY - 1):
            diff = cepstrum - self.cepstral_ring[i + 1]
            distances[i] = float(np.dot(diff, diff))
        self.cepstral_diffs = np.roll(self.cepstral_diffs, 1, axis=0)
        self.cepstral_diffs[0] = 0
        self.cepstral_diffs[0, 1:] = distances
        self.cepstral_diffs[1:, 0] = distances

        features = np.zeros(FEATURE_VECTOR_SIZE)
        # Higher bands cepstrum: bands 6..21
        features[NUM_LOWER_BANDS:NUM_BANDS] = cepstrum[NUM_LOWER_BANDS:]
        # Average and derivatives over the first 6 coefficients.
        curr = self.cepstral_ring[0]
        prev1 = self.cepstral_ring[1]
        prev2 = self.cepstral_ring[2]
        for i in range(NUM_LOWER_BANDS):
            features[i] = curr[i] + prev1[i] + prev2[i]
            features[NUM_BANDS + i] = curr[i] - prev2[i]
            features[NUM_BANDS + NUM_LOWER_BANDS + i] = curr[i] - 2 * prev1[i] + prev2[i]
        # Normalised cepstral correlation.
        cross = cross_correlation(ref_fft, lag_fft)
        for i in range(NUM_LOWER_BANDS):
            features[NUM_BANDS + 2 * NUM_LOWER_BANDS + i] = cross[i] / math.sqrt(0.001 + ref_energy[i] * lag_energy[i])
        # Variability.
        features[FEATURE_VECTOR_SIZE - 1] = _variability(self.cepstral_diffs)
        return False, features


def _variability(diffs):
    """ComputeVariability over the symmetric cepstral difference buffer."""
    total = 0.0
    count = 0
    for i in range(CEPSTRAL_HISTORY):
        for j in range(i + 1, CEPSTRAL_HISTORY):
            total += diffs[i, j]
            count += 1
    if count == 0:
        return 0.0
    return min(1.0, total / count)


def features_from_frames(frames_24k):
    """Runs the spectral part over 20 ms frames and returns (silence, features)."""
    extractor = SpectralFeatures()
    return extractor.compute(frames_24k[0], frames_24k[1])
