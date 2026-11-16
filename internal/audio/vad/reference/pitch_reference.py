"""Reference implementation of WebRTC's AGC2 VAD pitch estimation.

Translated from modules/audio_processing/agc2/rnn_vad/:
  lp_residual.cc, pitch_search.cc, pitch_search_internal.cc, auto_correlation.cc

Oracle for the Go port in internal/audio/vad, not production code.

Copyright (c) 2018 The WebRTC project authors. All Rights Reserved.
Use of this source code is governed by a BSD-style license that can be found in
the LICENSE file in the root of the source tree. An additional intellectual
property rights grant can be found in the PATENTS file. All contributing
project authors may be found in the AUTHORS file in the source tree.
"""

import math

import numpy as np

# common.h
SAMPLE_RATE_24K = 24000
FRAME_10MS_24K = 240
FRAME_20MS_24K = 480
MIN_PITCH_24K = SAMPLE_RATE_24K // 800          # 30
MAX_PITCH_24K = SAMPLE_RATE_24K // 62 + 1        # see below
MAX_PITCH_24K = 384                              # 24000 // 62.5
BUF_SIZE_24K = MAX_PITCH_24K + FRAME_20MS_24K    # 864
INITIAL_MIN_PITCH_24K = 3 * MIN_PITCH_24K        # 90
INITIAL_NUM_LAGS_24K = MAX_PITCH_24K - INITIAL_MIN_PITCH_24K   # 294
REFINE_NUM_LAGS_24K = MAX_PITCH_24K + 1          # 385
SAMPLE_RATE_12K = 12000
FRAME_20MS_12K = 240
BUF_SIZE_12K = BUF_SIZE_24K // 2
INITIAL_MIN_PITCH_12K = INITIAL_MIN_PITCH_24K // 2
MAX_PITCH_12K = MAX_PITCH_24K // 2
NUM_LAGS_12K = MAX_PITCH_12K - INITIAL_MIN_PITCH_12K
MIN_PITCH_48K = MIN_PITCH_24K * 2
MAX_PITCH_48K = MAX_PITCH_24K * 2

NUM_LPC_COEFFICIENTS = 5
SUB_HARMONIC_MULTIPLIERS = [3, 2, 3, 2, 5, 2, 3, 2, 3, 2, 5, 2, 3, 2]
INITIAL_PITCH_PERIOD_THRESHOLDS = [20, 45, 80, 125, 180, 245, 320, 405, 500, 605, 720, 845, 980, 1125]


def compute_auto_correlation_lpc(x):
    """ComputeAutoCorrelation over the first NUM_LPC_COEFFICIENTS lags."""
    out = np.zeros(NUM_LPC_COEFFICIENTS, dtype=np.float64)
    for lag in range(NUM_LPC_COEFFICIENTS):
        out[lag] = np.dot(x[: len(x) - lag], x[lag:])
    return out


def denoise_auto_correlation(auto_corr):
    """DenoiseAutoCorrelation: assume a -40 dB white noise floor."""
    auto_corr[0] *= 1.0001
    auto_corr[1] -= auto_corr[1] * 0.000064
    auto_corr[2] -= auto_corr[2] * 0.000256
    auto_corr[3] -= auto_corr[3] * 0.000576
    auto_corr[4] -= auto_corr[4] * 0.001024


def compute_initial_inverse_filter(auto_corr):
    """ComputeInitialInverseFilterCoefficients (Levinson-Durbin)."""
    lpc = np.zeros(NUM_LPC_COEFFICIENTS - 1, dtype=np.float64)
    error = auto_corr[0]
    for i in range(NUM_LPC_COEFFICIENTS - 1):
        reflection = 0.0
        for j in range(i):
            reflection += lpc[j] * auto_corr[i - j]
        reflection += auto_corr[i + 1]
        if abs(error) < 1e-6:
            error = math.copysign(1e-6, error)
        reflection /= -error
        lpc[i] = reflection
        for j in range((i + 1) >> 1):
            tmp1 = lpc[j]
            tmp2 = lpc[i - 1 - j]
            lpc[j] = tmp1 + reflection * tmp2
            lpc[i - 1 - j] = tmp2 + reflection * tmp1
        error -= reflection * reflection * error
        if error < 0.001 * auto_corr[0]:
            break
    return lpc


def compute_lpc_coefficients(x):
    """ComputeAndPostProcessLpcCoefficients."""
    auto_corr = compute_auto_correlation_lpc(x)
    if auto_corr[0] == 0.0:
        return np.zeros(NUM_LPC_COEFFICIENTS)
    denoise_auto_correlation(auto_corr)
    pre = compute_initial_inverse_filter(auto_corr)
    pre[0] *= 0.9
    pre[1] *= 0.9 * 0.9
    pre[2] *= 0.9 * 0.9 * 0.9
    pre[3] *= 0.9 * 0.9 * 0.9 * 0.9
    c = 0.8
    out = np.zeros(NUM_LPC_COEFFICIENTS)
    out[0] = pre[0] + c
    out[1] = pre[1] + c * pre[0]
    out[2] = pre[2] + c * pre[1]
    out[3] = pre[3] + c * pre[2]
    out[4] = c * pre[3]
    return out


def compute_lp_residual(lpc, x):
    """ComputeLpResidual."""
    y = np.zeros(len(x))
    y[0] = x[0]
    for i in range(1, NUM_LPC_COEFFICIENTS):
        y[i] = x[i] + float(np.dot(x[i - 1 :: -1][:i], lpc[:i]))
    for i in range(NUM_LPC_COEFFICIENTS, len(y)):
        y[i] = x[i] + float(np.dot(x[i - 1 : i - NUM_LPC_COEFFICIENTS : -1], lpc))
    return y


def decimate_2x(src):
    """Decimate2x."""
    return src[::2].copy()


def compute_auto_correlation_12k(pitch_buffer_12k):
    """AutoCorrelationCalculator::ComputeOnPitchBuffer.

    The C++ does this in the frequency domain with a 512-point FFT: it
    correlates the reversed reference frame against a chunk holding every
    sliding frame, then extracts the coefficients starting at
    kConvolutionLength - 1. The direct form below is the same operation.
    """
    convolution_length = BUF_SIZE_12K - MAX_PITCH_12K   # 240
    # The C++ reverses the reference frame (std::reverse_copy) and convolves it
    # with the chunk holding every sliding window, then extracts the result
    # starting at kConvolutionLength - 1. Reversing turns the convolution into
    # the correlation the inverted lags index.
    reference = pitch_buffer_12k[-convolution_length:][::-1]
    out = np.zeros(NUM_LAGS_12K, dtype=np.float64)
    for inverted_lag in range(NUM_LAGS_12K):
        window = pitch_buffer_12k[inverted_lag : inverted_lag + convolution_length]
        out[inverted_lag] = float(np.dot(window, reference))
    return out


def compute_auto_correlation_12k_lagged(pitch_buffer_12k):
    """Lagged-dot-product form of the 12 kHz auto-correlation.

    Kept beside the frequency-domain port above so the two can be compared; the
    inverted lag is measured from the maximum pitch period, so the lag for
    inverted_lag i is MAX_PITCH_12K - i.
    """
    out = np.zeros(NUM_LAGS_12K, dtype=np.float64)
    for inverted_lag in range(NUM_LAGS_12K):
        lag = MAX_PITCH_12K - inverted_lag
        out[inverted_lag] = float(
            np.dot(pitch_buffer_12k[: len(pitch_buffer_12k) - lag], pitch_buffer_12k[lag:]))
    return out


def compute_pitch_period_12k(pitch_buffer, auto_correlation):
    """ComputePitchPeriod12kHz."""
    frame = pitch_buffer[: FRAME_20MS_12K + 1]
    denominator = 1.0 + float(np.dot(frame, frame))
    best = [0, -1.0, 0.0]        # inverted lag, numerator, denominator
    second = [1, -1.0, 0.0]

    def stronger(a, b):
        return a[1] * b[2] > b[1] * a[2]

    for inverted_lag in range(NUM_LAGS_12K):
        if auto_correlation[inverted_lag] > 0.0:
            candidate = [inverted_lag, auto_correlation[inverted_lag] ** 2, denominator]
            if stronger(candidate, second):
                if stronger(candidate, best):
                    second = best
                    best = candidate
                else:
                    second = candidate
        y_old = pitch_buffer[inverted_lag]
        y_new = pitch_buffer[inverted_lag + FRAME_20MS_12K]
        denominator -= y_old * y_old
        denominator += y_new * y_new
        denominator = max(0.0, denominator)
    return best[0], second[0]


def compute_sliding_frame_square_energies_24k(pitch_buffer):
    """ComputeSlidingFrameSquareEnergies24kHz."""
    y_energy = np.zeros(REFINE_NUM_LAGS_24K, dtype=np.float64)
    yy = float(np.dot(pitch_buffer[:FRAME_20MS_24K], pitch_buffer[:FRAME_20MS_24K]))
    y_energy[0] = yy
    for inverted_lag in range(MAX_PITCH_24K):
        yy -= pitch_buffer[inverted_lag] * pitch_buffer[inverted_lag]
        yy += pitch_buffer[inverted_lag + FRAME_20MS_24K] * pitch_buffer[inverted_lag + FRAME_20MS_24K]
        yy = max(1.0, yy)
        y_energy[inverted_lag + 1] = yy
    return y_energy


def compute_auto_correlation_24k(inverted_lag, pitch_buffer):
    """ComputeAutoCorrelation from pitch_search_internal.cc.

    A fixed reference frame at kMaxPitch24kHz is correlated against a window
    sliding from the inverted lag, so the correlation peaks when the window
    reaches the reference - i.e. at the maximum pitch period.
    """
    reference = pitch_buffer[MAX_PITCH_24K : MAX_PITCH_24K + FRAME_20MS_24K]
    window = pitch_buffer[inverted_lag : inverted_lag + FRAME_20MS_24K]
    return float(np.dot(window, reference))


def get_pitch_pseudo_interpolation_offset(prev, curr, nxt):
    """GetPitchPseudoInterpolationOffset."""
    if (nxt - prev) > 0.7 * (curr - prev):
        return 1
    if (prev - nxt) > 0.7 * (curr - nxt):
        return -1
    return 0


PITCH_NEIGHBORHOOD_RADIUS = 2


def create_inverted_lag_range(inverted_lag):
    """CreateInvertedLagRange: a clipped +/-2 window around a candidate."""
    return (max(inverted_lag - PITCH_NEIGHBORHOOD_RADIUS, 0),
            min(inverted_lag + PITCH_NEIGHBORHOOD_RADIUS, INITIAL_NUM_LAGS_24K - 1))


def compute_pitch_period_48k(pitch_buffer, y_energy, pitch_candidates):
    """ComputePitchPeriod48kHz.

    Only the neighbours of the two 12 kHz candidates are searched, not the whole
    lag range.
    """
    best_candidate, second_candidate = pitch_candidates
    swap = best_candidate > second_candidate
    r1 = create_inverted_lag_range(second_candidate if swap else best_candidate)
    r2 = create_inverted_lag_range(best_candidate if swap else second_candidate)
    inverted_lags = list(range(r1[0], r1[1] + 1)) + list(range(r2[0], r2[1] + 1))
    best_lag = 0
    best_num = -1.0
    best_den = 0.0
    # The C++ leaves the array uninitialised and explicitly zeroes the two
    # boundary neighbours, which the pseudo-interpolation reads. Computing real
    # values there instead shifts the offset by one at the window edges.
    auto_corr = {lag: compute_auto_correlation_24k(lag, pitch_buffer) for lag in inverted_lags}
    low, high = min(inverted_lags), max(inverted_lags)
    if low > 0:
        auto_corr[low - 1] = 0.0
    if high < INITIAL_NUM_LAGS_24K - 1:
        auto_corr[high + 1] = 0.0
    for inverted_lag in inverted_lags:
        if auto_corr[inverted_lag] > 0.0:
            numerator = auto_corr[inverted_lag] ** 2
            denominator = y_energy[inverted_lag]
            if numerator * best_den > best_num * denominator:
                best_lag = inverted_lag
                best_num = numerator
                best_den = denominator
    if best_lag == 0 or best_lag >= INITIAL_NUM_LAGS_24K - 1:
        return best_lag * 2
    offset = get_pitch_pseudo_interpolation_offset(
        auto_corr.get(best_lag + 1, 0.0), auto_corr[best_lag], auto_corr.get(best_lag - 1, 0.0))
    return 2 * best_lag + offset


def get_alternative_pitch_period(period, multiplier, divisor):
    """GetAlternativePitchPeriod."""
    return (2 * multiplier * period + divisor) // (2 * divisor)


def is_alternative_stronger(last, initial, alternative, period_divisor):
    """IsAlternativePitchStrongerThanInitial."""
    lower_threshold_term = 0.0
    if abs(alternative[0] - last[0]) <= 1:
        lower_threshold_term = last[1]
    elif abs(alternative[0] - last[0]) == 2 and initial[0] > INITIAL_PITCH_PERIOD_THRESHOLDS[period_divisor - 2]:
        lower_threshold_term = 0.5 * last[1]
    threshold = max(0.3, 0.7 * initial[1] - lower_threshold_term)
    if alternative[0] < 3 * MIN_PITCH_24K:
        threshold = max(0.4, 0.85 * initial[1] - lower_threshold_term)
    elif alternative[0] < 2 * MIN_PITCH_24K:
        threshold = max(0.5, 0.9 * initial[1] - lower_threshold_term)
    return alternative[1] > threshold


def compute_extended_pitch_period_48k(pitch_buffer, y_energy, initial_period_48k, last_pitch_48k):
    """ComputeExtendedPitchPeriod48kHz."""
    x_energy = y_energy[MAX_PITCH_24K]

    def strength(xy, y_energy_value):
        return xy / math.sqrt(1.0 + x_energy * y_energy_value)

    best_period = min(initial_period_48k // 2, MAX_PITCH_24K - 1)
    best_xy = compute_auto_correlation_24k(MAX_PITCH_24K - best_period, pitch_buffer)
    best_y = y_energy[MAX_PITCH_24K - best_period]
    best_strength = strength(best_xy, best_y)
    initial = (best_period, best_strength)
    last = (last_pitch_48k[0] // 2, last_pitch_48k[1])

    max_divisor = (2 * initial[0]) // (2 * MIN_PITCH_24K - 1)
    for divisor in range(2, max_divisor + 1):
        period = get_alternative_pitch_period(initial[0], 1, divisor)
        dual = get_alternative_pitch_period(initial[0], SUB_HARMONIC_MULTIPLIERS[divisor - 2], divisor)
        if divisor == 2 and dual > MAX_PITCH_24K:
            dual = initial[0]
        xy_primary = compute_auto_correlation_24k(MAX_PITCH_24K - period, pitch_buffer)
        xy_secondary = compute_auto_correlation_24k(MAX_PITCH_24K - dual, pitch_buffer)
        xy = 0.5 * (xy_primary + xy_secondary)
        yy = 0.5 * (y_energy[MAX_PITCH_24K - period] + y_energy[MAX_PITCH_24K - dual])
        alternative = (period, strength(xy, yy))
        if is_alternative_stronger(last, initial, alternative, divisor):
            if alternative[1] > best_strength:
                best_period, best_strength = alternative
    return best_period * 2, best_strength


def estimate_pitch(pitch_buffer_24k):
    """PitchEstimator::Estimate, returning the 48 kHz pitch period."""
    buffer_12k = decimate_2x(pitch_buffer_24k)
    auto_corr_12k = compute_auto_correlation_12k(buffer_12k)
    best_12k, second_12k = compute_pitch_period_12k(buffer_12k, auto_corr_12k)
    best_12k *= 2
    second_12k *= 2
    y_energy = compute_sliding_frame_square_energies_24k(pitch_buffer_24k)
    period_48k = compute_pitch_period_48k(pitch_buffer_24k, y_energy, (best_12k, second_12k))
    extended, _ = compute_extended_pitch_period_48k(
        pitch_buffer_24k, y_energy, MAX_PITCH_48K - period_48k, (0, 0.0))
    return extended
