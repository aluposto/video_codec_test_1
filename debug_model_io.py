#!/usr/bin/env python3
"""
train/debug_model_io.py

Robust debug script to:
 - locate repo 'src' automatically
 - load a short clip (mp4 or frame-dir)
 - run one forward pass through DMCI and DMC
 - print returned keys/shapes/types/statistics

Usage:
  python train/debug_model_io.py --clip path/to/clip.mp4 --device cuda --frames 5
"""
# --- AUTO-LOCATE 'src' AND INSERT INTO sys.path ---
import os
import sys
THIS_FILE = os.path.abspath(__file__)
TRAIN_DIR = os.path.dirname(THIS_FILE)          # .../repo/train
REPO_ROOT = os.path.abspath(os.path.join(TRAIN_DIR, '..'))  # .../repo
candidates = [
    os.path.join(REPO_ROOT, 'src'),
    os.path.join(REPO_ROOT, 'DCVC', 'src'),
    os.path.join(REPO_ROOT, '..', 'src'),
]
found = False
for p in candidates:
    if os.path.isdir(p):
        if p not in sys.path:
            sys.path.insert(0, p)
        found = True
        break
if not found:
    for root, dirs, files in os.walk(REPO_ROOT):
        if 'models' in dirs:
            candidate_src = os.path.abspath(root)
            if candidate_src not in sys.path:
                sys.path.insert(0, candidate_src)
            found = True
            break
if not found:
    print("WARNING: could not auto-locate 'src' under repo root:", REPO_ROOT)
    print("sys.path (first entries):", sys.path[:6])
# --- end auto-locate ---

import argparse
import inspect
import traceback
from glob import glob

import cv2
import numpy as np
import torch
from torchvision.transforms import ToTensor

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def safe_instantiate(cls, cls_name="Model"):
    """
    Try to instantiate a class robustly:
    - try no-arg constructor
    - if fails, inspect signature and pass `None` for required params (best-effort)
    - print helpful diagnostics
    """
    try:
        inst = cls()
        print(f"[OK] Instantiated {cls_name}() with no args.")
        return inst
    except Exception as e_noarg:
        print(f"[WARN] {cls_name}() no-arg instantiation failed: {repr(e_noarg)}")
        # attempt to call with None for parameters without defaults
        try:
            sig = inspect.signature(cls)
            kwargs = {}
            for name, param in sig.parameters.items():
                # skip 'self' if present (for bound methods it isn't)
                if name == 'self':
                    continue
                if param.default is inspect._empty:
                    # required param: give None (best-effort)
                    kwargs[name] = None
            if kwargs:
                print(f"[INFO] Trying {cls_name} with kwargs: {kwargs}")
                inst = cls(**kwargs)
            else:
                inst = cls()
            print(f"[OK] Instantiated {cls_name} with fallback kwargs.")
            return inst
        except Exception as e_fallback:
            print(f"[ERROR] Failed to instantiate {cls_name} with fallback. Traceback below:")
            traceback.print_exc()
            raise RuntimeError(f"Could not instantiate {cls_name}: {e_fallback}") from e_fallback

def load_clip_frames(path_or_dir, max_frames=5):
    """
    If `path_or_dir` is a directory containing image files, load sorted image files.
    Otherwise treat as a video file and decode frames with OpenCV.
    Returns a list of numpy arrays (RGB uint8).
    """
    if os.path.isdir(path_or_dir):
        imgs = sorted(glob(os.path.join(path_or_dir, '*')))
        if len(imgs) == 0:
            raise RuntimeError(f"No images found in directory {path_or_dir}")
        arrs = []
        for p in imgs[:max_frames]:
            im = cv2.imread(p)
            if im is None:
                raise RuntimeError(f"cv2 could not read image: {p}")
            arrs.append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        return arrs

    # else video file
    if not os.path.isfile(path_or_dir):
        raise RuntimeError(f"Clip path does not exist: {path_or_dir}")

    cap = cv2.VideoCapture(path_or_dir)
    if not cap.isOpened():
        raise RuntimeError(f"cv2.VideoCapture failed to open: {path_or_dir}")
    frames = []
    read_count = 0
    while read_count < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        read_count += 1
    cap.release()
    if len(frames) == 0:
        raise RuntimeError(f"No frames decoded from video: {path_or_dir}")
    return frames

def tensor_stats(tensor):
    try:
        return {
            'shape': tuple(tensor.shape),
            'dtype': str(tensor.dtype),
            'min': float(torch.min(tensor).item()),
            'max': float(torch.max(tensor).item())
        }
    except Exception:
        return {'shape': None, 'dtype': None}

def pretty_print_output(out, prefix=""):
    """
    Print info about model output `out` which can be:
    - tensor
    - dict of tensors / nested dicts
    - other python objects
    """
    if torch.is_tensor(out):
        stats = tensor_stats(out)
        print(f"{prefix} (tensor) shape={stats['shape']}, dtype={stats['dtype']}, min={stats['min']:.6g}, max={stats['max']:.6g}")
        return

    if isinstance(out, dict):
        keys = list(out.keys())
        print(f"{prefix} (dict) keys: {keys}")
        for k, v in out.items():
            if torch.is_tensor(v):
                stats = tensor_stats(v)
                print(f"  {prefix}.{k}: tensor shape={stats['shape']}, dtype={stats['dtype']}, min={stats['min']:.6g}, max={stats['max']:.6g}")
            elif isinstance(v, dict):
                print(f"  {prefix}.{k}: (nested dict)")
                pretty_print_output(v, prefix=prefix + f".{k}")
            else:
                print(f"  {prefix}.{k}: type={type(v)} value={repr(v) if isinstance(v,(str,int,float)) else type(v)}")
        return

    # fallback
    print(f"{prefix} type={type(out)} repr={repr(out)[:200]}")

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--clip', required=True, help='Path to mp4 file or directory of frames')
    p.add_argument('--frames', type=int, default=5, help='Number of frames to load from clip')
    p.add_argument('--device', default='cuda', help='cuda or cpu')
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and 'cuda' in args.device else 'cpu')
    print(f"[INFO] Using device: {device}")

    # load clip frames
    print(f"[INFO] Loading up to {args.frames} frames from: {args.clip}")
    frames = load_clip_frames(args.clip, max_frames=args.frames)
    print(f"[INFO] Loaded {len(frames)} frames. Frame shapes (H,W,3):",
          [f.shape for f in frames])

    # convert to tensor: list of (C,H,W), values in [0,1] float
    to_tensor = ToTensor()
    try:
        clip_t = torch.stack([to_tensor(im) for im in frames], dim=0)  # T,C,H,W
    except Exception as e:
        print("[ERROR] Converting frames to tensors failed:", repr(e))
        raise

    # add batch dim
    clip_t = clip_t.unsqueeze(0)  # 1,T,C,H,W
    B, T, C, H, W = clip_t.shape
    print(f"[INFO] Clip tensor shape (B,T,C,H,W): {clip_t.shape}")

    # Import models
    try:
        from src.models.image_model import DMCI
        from src.models.video_model import DMC
    except Exception as e:
        print("[ERROR] Failed to import DMCI/DMC from src.models. Traceback:")
        traceback.print_exc()
        raise

    # instantiate image model
    try:
        image_model = safe_instantiate(DMCI, cls_name="DMCI")
    except Exception as e:
        print("[FATAL] Could not instantiate DMCI. Aborting.")
        raise

    # instantiate video model
    try:
        video_model = safe_instantiate(DMC, cls_name="DMC")
    except Exception as e:
        print("[FATAL] Could not instantiate DMC. Aborting.")
        raise

    # move to device and eval
    image_model.to(device).eval()
    video_model.to(device).eval()

    # Prepare input for image model: first frame (B,C,H,W)
    input_i = clip_t[:, 0].to(device)  # B,C,H,W
    print(f"[INFO] Running image model on tensor shape: {tuple(input_i.shape)}")

    # run image model
    try:
        with torch.no_grad():
            out_i = image_model(input_i)
    except Exception as e:
        print("[ERROR] image_model forward failed. Traceback:")
        traceback.print_exc()
        raise

    print("\n=== Image model output ===")
    try:
        pretty_print_output(out_i, prefix="image_out")
    except Exception as e:
        print("[WARN] pretty-print of image output failed:", repr(e))

    # Obtain a reconstructed tensor (try common keys)
    xhat_i = None
    lik_i = None
    if isinstance(out_i, dict):
        # common keys to try
        for key in ('x_hat', 'xhat', 'recon', 'reconstructed', 'reconstruction'):
            if key in out_i:
                xhat_i = out_i[key]
                break
        # find something that looks like likelihoods/probs
        for key in ('likelihoods', 'probs', 'probs_p', 'probs_y', 'likelihood_map', 'likelihood'):
            if key in out_i:
                lik_i = out_i[key]
                break
    elif torch.is_tensor(out_i):
        xhat_i = out_i

    if xhat_i is None:
        print("[WARN] Could not find reconstructed tensor in image model output (tried keys).")
    else:
        print("[INFO] image model reconstruction tensor found. Stats:")
        pretty_print_output(xhat_i, prefix="image_xhat")

    if lik_i is None:
        print("[INFO] No likelihoods found in image model output (that's OK for some models).")
    else:
        print("[INFO] image model likelihoods/probs found. Stats:")
        pretty_print_output(lik_i, prefix="image_likes")

    # Prepare P-frame input (second frame) and run video model using xhat_i as ref if available
    if T < 2:
        print("[INFO] Only single-frame clip provided; skipping video model forward.")
        return

    input_p = clip_t[:, 1].to(device)
    print(f"\n[INFO] Running video model on tensor shape: {tuple(input_p.shape)}")
    try:
        # try calling video_model with signature (cur, ref=prev) or (cur, prev) or (cur,) as fallback
        prev = xhat_i if (xhat_i is not None and torch.is_tensor(xhat_i)) else None
        with torch.no_grad():
            try:
                # preferred: cur, ref=prev
                out_p = video_model(input_p, ref=prev)
            except TypeError:
                try:
                    out_p = video_model(input_p, prev)
                except TypeError:
                    out_p = video_model(input_p)
    except Exception as e:
        print("[ERROR] video_model forward failed. Traceback:")
        traceback.print_exc()
        raise

    print("\n=== Video model output ===")
    try:
        pretty_print_output(out_p, prefix="video_out")
    except Exception as e:
        print("[WARN] pretty-print of video output failed:", repr(e))

    # Inspect expected fields
    xhat_p = None
    lik_p = None
    if isinstance(out_p, dict):
        for key in ('x_hat', 'xhat', 'recon', 'reconstructed', 'reconstruction'):
            if key in out_p:
                xhat_p = out_p[key]
                break
        for key in ('likelihoods', 'probs', 'probs_p', 'probs_y', 'likelihood_map', 'likelihood'):
            if key in out_p:
                lik_p = out_p[key]
                break
    elif torch.is_tensor(out_p):
        xhat_p = out_p

    if xhat_p is None:
        print("[WARN] Could not find reconstructed tensor in video model output (tried common keys).")
    else:
        print("[INFO] video model reconstruction tensor found. Stats:")
        pretty_print_output(xhat_p, prefix="video_xhat")

    if lik_p is None:
        print("[INFO] No likelihoods found in video model output (that's OK for some models).")
    else:
        print("[INFO] video model likelihoods/probs found. Stats:")
        pretty_print_output(lik_p, prefix="video_likes")

    print("\n[OK] debug_model_io completed successfully.")

if __name__ == '__main__':
    main()
