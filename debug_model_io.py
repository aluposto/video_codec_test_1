#!/usr/bin/env python3
"""
Debug script: load one small clip and run through DMCI and DMC to print output keys/shapes.
Usage:
  PYTHONPATH=$PWD/src python train/debug_model_io.py --clip data/sample/clip_0001 --device cuda
"""
import argparse
import os
import cv2
import torch
from torchvision.transforms import ToTensor
from glob import glob
from src.models.image_model import DMCI
from src.models.video_model import DMC

def load_clip_frames(dir_or_mp4, frames=5):
    # if it's a directory with images:
    if os.path.isdir(dir_or_mp4):
        imgs = sorted(glob(os.path.join(dir_or_mp4, '*')))
        imgs = imgs[:frames]
        arrs = [cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB) for p in imgs]
        return arrs
    # else try to read video:
    cap = cv2.VideoCapture(dir_or_mp4)
    arrs = []
    while len(arrs) < frames:
        ok, img = cap.read()
        if not ok:
            break
        arrs.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    cap.release()
    return arrs

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--clip', required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--frames', type=int, default=5)
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    imgs = load_clip_frames(args.clip, frames=args.frames)
    if len(imgs) < args.frames:
        raise RuntimeError("Not enough frames found in clip")

    to_tensor = ToTensor()
    clip_t = torch.stack([to_tensor(im) for im in imgs], dim=0)  # T,C,H,W
    clip_t = clip_t.unsqueeze(0).to(device)  # 1,T,C,H,W for batch dimension

    model_i = DMCI().to(device).eval()
    model_p = DMC().to(device).eval()

    with torch.no_grad():
        # I-frame
        i_frame = clip_t[0,0]  # C,H,W
        print("I-frame shape:", i_frame.shape)
        out_i = model_i(i_frame.unsqueeze(0))  # B=1
        print("Image model output keys:", list(out_i.keys()))
        for k,v in out_i.items():
            if torch.is_tensor(v):
                print(f"  {k}: tensor shape {v.shape}, dtype {v.dtype}, min/max {v.min().item():.3g}/{v.max().item():.3g}")
            else:
                print(f"  {k}: type {type(v)}")

        # P-frame (one step)
        prev = out_i.get('x_hat', out_i.get('recon', None))
        if prev is None:
            print("WARNING: image model did not return 'x_hat' or 'recon'. Adapt code to your model.")
        cur = clip_t[0,1]
        out_p = model_p(cur.unsqueeze(0), ref=prev)
        print("Video model output keys:", list(out_p.keys()))
        for k,v in out_p.items():
            if torch.is_tensor(v):
                print(f"  {k}: tensor shape {v.shape}, dtype {v.dtype}, min/max {v.min().item():.3g}/{v.max().item():.3g}")
            else:
                print(f"  {k}: type {type(v)}")

if __name__ == '__main__':
    main()
