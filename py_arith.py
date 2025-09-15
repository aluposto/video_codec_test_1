# src/py_arith.py
# Pure-Python integer range coder designed to encode symbol sequences
# using per-symbol PMFs (probability mass functions).
#
# Requirements: numpy, torch (torch only for convenience; encoder/decoder operate on CPU numpy arrays)
#
# Notes:
# - This is NOT optimized for C++ speed. It uses vectorized numpy ops and searchsorted
#   to make decoding faster than pure Python loops.
# - The encoder expects sequences of discrete integer symbols and corresponding PMFs
#   (one PMF per symbol) which sum to 1.0 (floating). We convert PMFs to integer
#   cumulative frequencies using a fixed precision (TOTAL).
#
# Usage functions:
# - encode_pmfs_symbols(pmfs_list, symbols_list, total=1<<16) -> bytes
# - decode_pmfs_bytes(pmfs_list, bitstream_bytes, total=1<<16) -> symbols_list
#
# pmfs_list: list (or iterable) of 1D numpy arrays (shape [K]) or torch tensors
# symbols_list: list or 1D numpy array of ints in [0, K-1]
#
# The decoder requires the same pmfs_list in the same order as the encoder.
# This fits DCVC usage where both encoder and decoder produce the same context-based pmfs.

import numpy as np
import math
from typing import Sequence, List, Tuple, Union
import io

# Range coder constants (32-bit state)
MASK32 = (1 << 32) - 1
TOP = 1 << 24
BOTTOM = 1 << 16

class RangeEncoder:
    def __init__(self):
        self.low = 0
        self.range = MASK32
        self.buffer = bytearray()

    def _normalize(self):
        while self.range < BOTTOM:
            # output top byte of low
            self.buffer.append((self.low >> 24) & 0xFF)
            self.low = (self.low << 8) & MASK32
            self.range = (self.range << 8) & MASK32

    def encode_symbol(self, cum, freq, tot):
        # cum, freq, tot are integers
        r = self.range // tot
        self.low = (self.low + r * cum) & MASK32
        self.range = (r * freq) & MASK32
        self._normalize()

    def finish(self):
        # flush 4 bytes of the final low value (big-endian)
        for _ in range(4):
            self.buffer.append((self.low >> 24) & 0xFF)
            self.low = (self.low << 8) & MASK32
        return bytes(self.buffer)

class RangeDecoder:
    def __init__(self, data: bytes):
        self.data = data
        self.pos = 0
        self.code = 0
        self.range = MASK32
        # initialize code with first 4 bytes
        for _ in range(4):
            self.code = ((self.code << 8) | self._read_byte()) & MASK32

    def _read_byte(self):
        if self.pos >= len(self.data):
            return 0
        b = self.data[self.pos]
        self.pos += 1
        return b

    def get_target(self, tot):
        # return integer target in [0, tot-1]
        r = self.range // tot
        if r == 0:
            return 0
        return min((self.code // r), tot - 1)

    def remove_symbol(self, cum, freq, tot):
        r = self.range // tot
        self.code = (self.code - r * cum) & MASK32
        self.range = (r * freq) & MASK32
        # renormalize
        while self.range < BOTTOM:
            self.range = (self.range << 8) & MASK32
            self.code = ((self.code << 8) | self._read_byte()) & MASK32

# Helper: convert a PMF (float array) to integer cumulative frequencies
def pmf_to_cumfreq(pmf: np.ndarray, total: int) -> Tuple[np.ndarray, int]:
    """
    pmf: 1D numpy array of non-negative floats summing ≈1.0
    total: integer total frequency (e.g., 2**16)
    Returns: cum (length K+1) of ints where cum[0]=0, cum[K]=total
             and freq_i = cum[i+1] - cum[i] >= 1 for stability
    """
    if not isinstance(pmf, np.ndarray):
        pmf = np.asarray(pmf, dtype=np.float64)
    # avoid zeros entirely: add small epsilon proportional to pmf sum
    eps = 1e-
