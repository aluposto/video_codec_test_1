#!/usr/bin/env python3
"""
train/train_video.py
Single-file training loop for DMCI (I-frame) + DMC (P-frames).
Usage example:
  export PYTHONPATH=$PWD/src:$PYTHONPATH
  python train/train_video.py \
    --train_root data/train \
    --val_root data/val \
    --save_dir runs/gaming_exp1 \
    --batch_size 4 \
    --frames 5 \
    --crop 256 \
    --epochs 50 \
    --mixed_precision
"""
import argparse
import os
import random
import math
from glob import glob
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
from tqdm import tqdm
from torch import nn
from torch.cuda import amp

from src.models.image_model import DMCI
from src.models.video_model import DMC

# ---------------- Dataset ----------------
class ClipDataset(Dataset):
    def __init__(self, root, frames=5, crop=256, mode='train'):
        self.root = Path(root)
        self.frames = frames
        self.crop = crop
        self.mode = mode

        mp4s = sorted(self.root.glob('**/*.mp4'))
        if mp4s:
            self.sources = [str(p) for p in mp4s]
            self.kind = 'video'
        else:
            dirs = [d for d in self.root.iterdir() if d.is_dir()]
            if not dirs:
                raise RuntimeError(f"No video files or frame dirs in {root}")
            self.sources = [str(d) for d in dirs]
            self.kind = 'frames'
        self.to_tensor = ToTensor()

    def read_frames_from_video(self, path):
        cap = cv2.VideoCapture(path)
        frames = []
        ok, img = cap.read()
        while ok:
            frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ok, img = cap.read()
        cap.release()
        return frames

    def read_frames_from_dir(self, d):
        imgs = sorted(glob(str(Path(d) / '*')))
        return [cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB) for p in imgs]

    def __len__(self):
        return max(len(self.sources) * 100, len(self.sources))

    def __getitem__(self, idx):
        src = random.choice(self.sources)
        if self.kind == 'video':
            frames = self.read_frames_from_video(src)
        else:
            frames = self.read_frames_from_dir(src)

        if len(frames) < self.frames:
            frames = frames + [frames[-1]] * (self.frames - len(frames))

        start = random.randint(0, max(0, len(frames) - self.frames))
        clip = frames[start:start + self.frames]
        clip_t = [self.to_tensor(im) for im in clip]  # list of C,H,W
        clip_t = torch.stack(clip_t, dim=0)  # T,C,H,W

        # resize if smaller than crop
        _, C, H, W = clip_t.shape
        if H < self.crop or W < self.crop:
            scale = max(self.crop / H, self.crop / W)
            new_h, new_w = math.ceil(H * scale), math.ceil(W * scale)
            clip_t = F.interpolate(clip_t, size=(new_h, new_w), mode='bilinear', align_corners=False)
            _, C, H, W = clip_t.shape

        top = random.randint(0, H - self.crop)
        left = random.randint(0, W - self.crop)
        clip_t = clip_t[:, :, top:top + self.crop, left:left + self.crop]

        if self.mode == 'train' and random.random() < 0.5:
            clip_t = torch.flip(clip_t, dims=[3])  # horizontal flip

        return clip_t  # T,C,H,W

# ---------------- Helpers ----------------
def bits_per_pixel_from_likelihoods(lik_dict, pixels):
    total_bits = 0.0
    for v in lik_dict.values():
        if not torch.is_tensor(v):
            continue
        total_bits += (-torch.log2(torch.clamp(v, min=1e-12))).sum()
    bpp = total_bits / float(pixels)
    return bpp

def psnr_batch(x, x_hat, data_range=1.0):
    mse = F.mse_loss(x_hat, x, reduction='none')
    mse = mse.view(mse.size(0), -1).mean(dim=1)
    psnr = 10.0 * torch.log10((data_range ** 2) / mse)
    return psnr.mean().item()

def collate_fn(batch):
    # batch: list of T,C,H,W -> stack to B,T,C,H,W
    return torch.stack(batch, dim=0)

# ---------------- Training & Validation ----------------
def validate(image_model, video_model, val_loader, device, args):
    image_model.eval(); video_model.eval()
    tot_psnr = 0.0; tot_bpp = 0.0; n = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='val', leave=False):
            batch = batch.to(device)  # B,T,C,H,W
            B, T, C, H, W = batch.shape
            i_frames = batch[:,0]  # B,C,H,W
            p_frames = [batch[:, t] for t in range(1, T)]

            out_i = image_model(i_frames)
            xhat_i = out_i.get('x_hat', out_i.get('recon', None))
            lik_i = out_i.get('likelihoods', out_i.get('probs', {}))
            if xhat_i is None:
                raise RuntimeError("Image model did not return expected key 'x_hat' or 'recon'.")

            total_lik = {}
            if isinstance(lik_i, dict):
                for k,v in lik_i.items(): total_lik[f'i_{k}'] = v
            else:
                total_lik['i'] = lik_i

            prev = xhat_i
            recons = [xhat_i]
            for t, cur in enumerate(p_frames):
                out_p = video_model(cur, ref=prev)
                xhat_p = out_p.get('x_hat', out_p.get('recon', None))
                lik_p = out_p.get('likelihoods', out_p.get('probs', {}))
                if xhat_p is None:
                    raise RuntimeError("Video model did not return expected key 'x_hat' or 'recon'.")
                recons.append(xhat_p)
                prev = xhat_p
                if isinstance(lik_p, dict):
                    for k,v in lik_p.items(): total_lik[f'p_{t}_{k}'] = v
                else:
                    total_lik[f'p_{t}'] = lik_p

            recon_stack = torch.stack(recons, dim=0).permute(1,0,2,3,4).reshape(-1, C, H, W)
            target_stack = batch.permute(0,1,2,3,4).reshape(-1, C, H, W)
            psnr_val = psnr_batch(target_stack, recon_stack)
            pixels = recon_stack.size(0) * H * W
            bpp = bits_per_pixel_from_likelihoods(total_lik, pixels)
            tot_psnr += psnr_val; tot_bpp += bpp; n += 1
    return tot_psnr / n, (tot_bpp / n).item()

def rd_loss(recon_flat, target_flat, total_lik, lambda_rd):
    distortion = F.mse_loss(recon_flat, target_flat)
    Bn, C, H, W = target_flat.shape
    pixels = Bn * H * W
    rate_bpp = bits_per_pixel_from_likelihoods(total_lik, pixels)
    total = lambda_rd * distortion + rate_bpp
    return total, distortion.detach(), rate_bpp.detach()

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--train_root', required=True)
    p.add_argument('--val_root', required=True)
    p.add_argument('--save_dir', default='./checkpoints')
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--frames', type=int, default=5)
    p.add_argument('--crop', type=int, default=256)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--lambda_rd', type=float, default=0.01)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--mixed_precision', action='store_true')
    p.add_argument('--device', default='cuda')
    p.add_argument('--resume', default=None)
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # reproducibility seeds
    random.seed(42); np.random.seed(42); torch.manual_seed(42)

    # models
    image_model = DMCI().to(device)
    video_model = DMC().to(device)

    train_ds = ClipDataset(args.train_root, frames=args.frames, crop=args.crop, mode='train')
    val_ds = ClipDataset(args.val_root, frames=args.frames, crop=args.crop, mode='val')

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=max(1, args.batch_size//2), shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)

    optimizer = torch.optim.Adam(list(image_model.parameters()) + list(video_model.parameters()), lr=args.lr)
    scaler = amp.GradScaler(enabled=args.mixed_precision)

    start_epoch = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        image_model.load_state_dict(ck['image'])
        video_model.load_state_dict(ck['video'])
        optimizer.load_state_dict(ck.get('optim', optimizer.state_dict()))
        start_epoch = ck.get('epoch', 0) + 1
        print("Resumed from", args.resume)

    for epoch in range(start_epoch, args.epochs):
        image_model.train(); video_model.train()
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}', leave=False)
        epoch_loss = 0.0
        for batch in pbar:
            batch = batch.to(device)  # B,T,C,H,W
            B, T, C, H, W = batch.shape
            i_frame = batch[:,0]  # B,C,H,W
            p_frames = [batch[:, t] for t in range(1, T)]

            optimizer.zero_grad()
            with amp.autocast(enabled=args.mixed_precision):
                out_i = image_model(i_frame)
                xhat_i = out_i.get('x_hat', out_i.get('recon', None))
                lik_i = out_i.get('likelihoods', out_i.get('probs', {}))
                if xhat_i is None:
                    raise RuntimeError("Image model missing 'x_hat' or 'recon' key.")

                total_lik = {}
                if isinstance(lik_i, dict):
                    for k,v in lik_i.items(): total_lik[f'i_{k}'] = v
                else:
                    total_lik['i'] = lik_i

                prev = xhat_i
                recons = [xhat_i]
                for t, cur in enumerate(p_frames):
                    out_p = video_model(cur, ref=prev)
                    xhat_p = out_p.get('x_hat', out_p.get('recon', None))
                    lik_p = out_p.get('likelihoods', out_p.get('probs', {}))
                    if xhat_p is None:
                        raise RuntimeError("Video model missing 'x_hat' or 'recon' key.")
                    recons.append(xhat_p)
                    prev = xhat_p
                    if isinstance(lik_p, dict):
                        for k,v in lik_p.items(): total_lik[f'p_{t}_{k}'] = v
                    else:
                        total_lik[f'p_{t}'] = lik_p

                recon_stack = torch.stack(recons, dim=0).permute(1,0,2,3,4).reshape(-1, C, H, W)
                target_stack = batch.permute(0,1,2,3,4).reshape(-1, C, H, W)

                loss, dist, bpp = rd_loss(recon_stack, target_stack, total_lik, args.lambda_rd)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(list(image_model.parameters()) + list(video_model.parameters()), 5.0)
            scaler.step(optimizer); scaler.update()

            epoch_loss += loss.item()
            pbar.set_postfix({'loss': epoch_loss / (pbar.n + 1), 'dist': float(dist), 'bpp': float(bpp)})

        # save checkpoint
        torch.save({'epoch': epoch, 'image': image_model.state_dict(),
                    'video': video_model.state_dict(), 'optim': optimizer.state_dict()},
                   os.path.join(args.save_dir, f'ckpt_epoch_{epoch:04d}.pth'))
        print(f"Saved ckpt_epoch_{epoch:04d}")

        # validate
        val_psnr, val_bpp = validate(image_model, video_model, val_loader, device, args)
        print(f"[Epoch {epoch}] VAL PSNR: {val_psnr:.3f} dB    VAL bpp: {val_bpp:.6f}")

if __name__ == '__main__':
    main()
