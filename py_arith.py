# src/py_arith.py
import numpy as np
from typing import Sequence, List, Union, Tuple

MASK32 = (1 << 32) - 1
BOTTOM = 1 << 16

class RangeEncoder:
    def __init__(self):
        self.low = 0
        self.range = MASK32
        self.buffer = bytearray()
    def _normalize(self):
        while self.range < BOTTOM:
            self.buffer.append((self.low >> 24) & 0xFF)
            self.low = (self.low << 8) & MASK32
            self.range = (self.range << 8) & MASK32
    def encode_symbol(self, cum, freq, tot):
        r = self.range // tot
        self.low = (self.low + r * cum) & MASK32
        self.range = (r * freq) & MASK32
        self._normalize()
    def finish(self):
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
        for _ in range(4):
            self.code = ((self.code << 8) | self._read_byte()) & MASK32
    def _read_byte(self):
        if self.pos >= len(self.data): return 0
        b = self.data[self.pos]; self.pos += 1; return b
    def get_target(self, tot):
        r = self.range // tot
        if r == 0: return 0
        return min((self.code // r), tot - 1)
    def remove_symbol(self, cum, freq, tot):
        r = self.range // tot
        self.code = (self.code - r * cum) & MASK32
        self.range = (r * freq) & MASK32
        while self.range < BOTTOM:
            self.range = (self.range << 8) & MASK32
            self.code = ((self.code << 8) | self._read_byte()) & MASK32

def pmf_to_cumfreq(pmf: np.ndarray, total: int) -> Tuple[np.ndarray,int]:
    if not isinstance(pmf, np.ndarray):
        pmf = np.asarray(pmf, dtype=np.float64)
    pmf = np.maximum(pmf, 0.0)
    s = pmf.sum()
    if s <= 0.0:
        K = pmf.shape[0]; freqs = np.ones(K, dtype=np.int64)
        cum = np.concatenate(([0], np.cumsum(freqs))); return cum, int(cum[-1])
    pmf = pmf / s
    raw = pmf * (total - 1)
    freqs = np.floor(raw).astype(np.int64)
    zeros = np.where(freqs == 0)[0]
    for z in zeros: freqs[z] += 1
    cur_total = int(freqs.sum()); diff = int(total - cur_total)
    if diff > 0:
        idx = np.argsort(-pmf)[:diff]; freqs[idx] += 1
    elif diff < 0:
        idx = np.argsort(pmf)[:(-diff)]
        for i in idx:
            if freqs[i] > 1: freqs[i] -= 1
    cum = np.concatenate(([0], np.cumsum(freqs)))
    return cum, int(cum[-1])

def encode_pmfs_symbols(pmfs: Sequence[Union[np.ndarray,'torch.Tensor']],
                        symbols: Sequence[int],
                        total: int = 1<<16) -> bytes:
    enc = RangeEncoder()
    for pmf, sym in zip(pmfs, symbols):
        if 'torch' in str(type(pmf)): pmf = pmf.cpu().numpy()
        cum, tot = pmf_to_cumfreq(np.asarray(pmf, dtype=np.float64), total)
        freq = cum[sym+1] - cum[sym]
        if freq <= 0:
            K = len(cum)-1
            cum, tot = pmf_to_cumfreq(np.ones(K,dtype=np.float64)/K, total)
            freq = cum[sym+1] - cum[sym]
        enc.encode_symbol(int(cum[sym]), int(freq), int(tot))
    return enc.finish()

def decode_pmfs_bytes(pmfs: Sequence[Union[np.ndarray,'torch.Tensor']],
                      data: bytes, total: int = 1<<16) -> List[int]:
    dec = RangeDecoder(data); out = []
    for pmf in pmfs:
        if 'torch' in str(type(pmf)): pmf = pmf.cpu().numpy()
        cum, tot = pmf_to_cumfreq(np.asarray(pmf, dtype=np.float64), total)
        target = dec.get_target(int(tot))
        s = np.searchsorted(cum, target, side='right') - 1
        if s < 0: s = 0
        if s >= len(cum) - 1: s = len(cum) - 2
        freq = cum[s+1] - cum[s]
        dec.remove_symbol(int(cum[s]), int(freq), int(tot))
        out.append(int(s))
    return out

def encode_batch_pmfs_symbols(pmfs_batch, symbols_batch, total: int = 1<<16) -> bytes:
    if 'torch' in str(type(pmfs_batch)):
        import torch as _torch; pmfs_batch = pmfs_batch.detach().cpu().numpy()
    if 'torch' in str(type(symbols_batch)):
        symbols_batch = symbols_batch.detach().cpu().numpy().astype(np.int64)
    pmfs_list = (pmfs_batch[i] for i in range(pmfs_batch.shape[0]))
    return encode_pmfs_symbols(pmfs_list, symbols_batch.tolist(), total=total)

def decode_batch_pmfs_bytes(pmfs_batch, data, total: int = 1<<16):
    if 'torch' in str(type(pmfs_batch)):
        import torch as _torch; pmfs_batch = pmfs_batch.detach().cpu().numpy()
    pmfs_list = (pmfs_batch[i] for i in range(pmfs_batch.shape[0]))
    out = decode_pmfs_bytes(pmfs_list, data, total=total)
    import numpy as _np; return _np.array(out, dtype=_np.int64)
