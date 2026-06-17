"""
Verify C program hex files for layer 20.
Replicates the C program's forward_convolutional_layer_q() computation exactly.
"""
import numpy as np
import os

BASE = r"C:\skeleton\bin"
LOG_FEAMAP = os.path.join(BASE, "log_feamap")
LOG_PARAM = os.path.join(BASE, "log_param")

# Layer 20 parameters from aix2024.cfg
# [convolutional] size=1 stride=1 pad=1 filters=195 activation=linear
# Input from ROUTE(-1,8): upsample(128ch, 16x16) + CONV08(256ch, 16x16) = 384ch, 16x16
L_N = 195      # output channels
L_C = 384      # input channels
L_H = 16       # input height
L_W = 16       # input width
L_SIZE = 1     # kernel size
L_STRIDE = 1
L_PAD = 0      # padding = size/2 = 1/2 = 0 (integer division!)
OUT_H = 16     # (16 + 2*0 - 1)/1 + 1 = 16
OUT_W = 16
OUT_SIZE = OUT_H * OUT_W

INPUT_QM = 16
WEIGHT_QM = 64
BIAS_MULT = 1024    # input_qm * weight_qm
NEXT_INPUT_QM = 1   # last CONV layer

MAX_VAL_8 = 127
MAX_VAL_16 = 32767
MAX_VAL_32 = 2147483647
MAX_VAL_UINT_8 = 255

def max_abs(src, max_val):
    """Exact C replica: if abs(src) > abs(max_val), return (src>0)?max_val:-max_val-1"""
    if abs(src) > abs(max_val):
        return max_val if src > 0 else -max_val - 1
    return src

# Load hex files
def load_hex_uint8(filepath):
    with open(filepath, 'r') as f:
        lines = [l.strip() for l in f if l.strip()]
    return np.array([int(l, 16) for l in lines], dtype=np.uint8)

def load_hex_int8(filepath):
    with open(filepath, 'r') as f:
        lines = [l.strip() for l in f if l.strip()]
    vals = np.array([int(l, 16) for l in lines], dtype=np.uint8)
    return vals.view(np.int8)

def load_hex_int16(filepath):
    with open(filepath, 'r') as f:
        lines = [l.strip() for l in f if l.strip()]
    vals = np.array([int(l, 16) for l in lines], dtype=np.uint16)
    return vals.view(np.int16)

print("Loading hex files...")
ifm_hex = load_hex_uint8(os.path.join(LOG_FEAMAP, "CONV20_input.hex"))
weights_hex = load_hex_int8(os.path.join(LOG_PARAM, "CONV20_param_weight.hex"))
biases_hex = load_hex_int16(os.path.join(LOG_PARAM, "CONV20_param_biases.hex"))
scales_hex = load_hex_uint8(os.path.join(LOG_PARAM, "CONV20_param_scales.hex"))
ofm_hex = load_hex_uint8(os.path.join(LOG_FEAMAP, "CONV20_output.hex"))

print(f"IFM: {ifm_hex.shape[0]} values (expected {L_C * L_H * L_W})")
print(f"Weights: {weights_hex.shape[0]} values (expected {L_N * L_SIZE * L_SIZE * L_C})")
print(f"Biases: {biases_hex.shape[0]} values (expected {L_N})")
print(f"Scales: {scales_hex.shape[0]} values (expected {L_N})")
print(f"OFM: {ofm_hex.shape[0]} values (expected {L_N * OUT_SIZE})")

# Reshape IFM: [C, H, W], interpret as int8
ifm = ifm_hex.reshape(L_C, L_H, L_W).astype(np.int32)
ifm = np.where(ifm > 127, ifm - 256, ifm).astype(np.int32)

# Reshape weights: [N, K] where K = size*size*C
K = L_SIZE * L_SIZE * L_C
weights = weights_hex.reshape(L_N, K).astype(np.int32)
biases = biases_hex.astype(np.int32)

# im2col: output [channels_col, height_col, width_col]
# channels_col = C * size * size, height_col = out_h, width_col = out_w
print("\nPerforming im2col...")
channels_col = L_C * L_SIZE * L_SIZE
data_col = np.zeros((channels_col, OUT_H, OUT_W), dtype=np.int32)

for c in range(channels_col):
    w_offset = c % L_SIZE
    h_offset = (c // L_SIZE) % L_SIZE
    c_im = c // L_SIZE // L_SIZE
    for h in range(OUT_H):
        for w in range(OUT_W):
            im_row = h_offset + h * L_STRIDE
            im_col = w_offset + w * L_STRIDE
            row = im_row - L_PAD
            col = im_col - L_PAD
            if row < 0 or col < 0 or row >= L_H or col >= L_W:
                data_col[c, h, w] = 0
            else:
                data_col[c, h, w] = ifm[c_im, row, col]

# Flatten to [K, out_size]
data_col_flat = data_col.reshape(channels_col, -1).astype(np.int32)

# GEMM: exact replica of gemm_nn_int8_int32
# M=1, N=out_size, K=K, ALPHA=1
# For each filter t: output_q[t*n : t*n+n] = weights[t] @ data_col
print("Performing GEMM (int32 accumulation)...")
output_q = np.zeros((L_N, OUT_SIZE), dtype=np.int32)
for fil in range(L_N):
    c_tmp = np.zeros(OUT_SIZE, dtype=np.int32)
    for k in range(K):
        A_PART = weights[fil, k]  # ALPHA=1, so A_PART = A[i*lda+k]
        for j in range(OUT_SIZE):
            c_tmp[j] += int(A_PART) * int(data_col_flat[k, j])
    # Per-row max_abs clamping with MAX_VAL_32
    for j in range(OUT_SIZE):
        output_q[fil, j] = max_abs(c_tmp[j], MAX_VAL_32)

# Add biases
print("Adding biases...")
for fil in range(L_N):
    for j in range(OUT_SIZE):
        output_q[fil, j] += biases[fil]

# Activation: linear (no ReLU for layer 20)

# Dequantize: l.output[i] = output_q[i] * ALPHA1
print("Dequantizing...")
ALPHA1 = 1.0 / (INPUT_QM * WEIGHT_QM)  # 1/1024
output_float = output_q.astype(np.float64) * ALPHA1

# Requantize to uint8 OFM
# C code: int16_t src = l.output[i] * next_input_quant_multiplier; (TRUNCATION!)
#          uint8_t pixel = max_abs(src, MAX_VAL_UINT_8);
print("Requantizing to uint8 OFM...")
ofm_computed = np.zeros((L_N, OUT_SIZE), dtype=np.int32)
for chn in range(L_N):
    for idx in range(OUT_SIZE):
        src = int(output_float[chn, idx] * NEXT_INPUT_QM)  # float->int16 truncation
        val = max_abs(src, MAX_VAL_UINT_8)
        ofm_computed[chn, idx] = val & 0xFF  # uint8_t cast

ofm_computed_uint8 = ofm_computed.astype(np.uint8)

# Compare
ofm_hex_reshaped = ofm_hex.reshape(L_N, OUT_SIZE)
ofm_match = np.all(ofm_computed_uint8.flatten() == ofm_hex)
print(f"\nOFM match: {ofm_match}")
if not ofm_match:
    mismatches = np.where(ofm_computed_uint8.flatten() != ofm_hex)
    print(f"Number of OFM mismatches: {len(mismatches[0])}")
    for i in range(min(30, len(mismatches[0]))):
        idx = mismatches[0][i]
        chn = idx // OUT_SIZE
        pos = idx % OUT_SIZE
        col = pos % OUT_W
        row = pos // OUT_W
        print(f"  Ch={chn}, Col={col}, Row={row}: computed={ofm_computed_uint8.flatten()[idx]:02x}, hex={ofm_hex[idx]:02x}")

# Compare output_q vs RTL Expected
rtl_expected_col0_row0 = {
    0: 0, 1: 0, 2: -1, 3: 0,
    4: 0, 5: 0, 6: 0, 7: 0,
    8: -1, 9: 0, 10: -1, 11: 0,
    12: 0, 13: 0, 14: 1, 15: 0,
}

print("\n=== Comparing output_q (post-bias) vs RTL Expected at Col=0,Row=0 ===")
pos_idx = 0
for ch in range(16):
    computed = output_q[ch, pos_idx]
    expected = rtl_expected_col0_row0.get(ch, '?')
    match = "✓" if computed == expected else "✗"
    print(f"  Ch={ch:3d}: computed={computed:8d}, RTL Expected={expected:>4}, {match}")

print("\n=== Comparing OFM uint8 (as signed) vs RTL Expected ===")
for ch in range(16):
    computed = ofm_computed_uint8[ch, pos_idx]
    expected = rtl_expected_col0_row0.get(ch, '?')
    match = "✓" if computed == expected else "✗"
    print(f"  Ch={ch:3d}: OFM uint8={int(computed):4d}, RTL Expected={expected:>4}, {match}")

print("\nDone.")