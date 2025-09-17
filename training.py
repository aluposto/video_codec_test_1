
import sys, os
# ensure repo root is on PYTHONPATH so 'src' package can be imported when running as a script
repo_root = os.path.dirname(os.path.abspath(__file__))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

#!/usr/bin/env python3
"""
training.py - Training loop for DCVC-style models on pre-extracted frames.

Usage (single GPU):
  python training.py --train-glob "data/frames/train/**/*.png" --val-glob "data/frames/val/**/*.png" \
    --patch-size 256 --temporal-len 4 --batch-size 6 --epochs 30 --cuda --amp \
    --pretrained ./checkpoints/cvpr2025_video.pth.tar --lambda-rd 0.01 --save-dir ./checkpoints_finetune

If your repo's VideoModel constructor signature differs, edit the import/constructor near top.
"""
import argparse, os, math, random, time
from glob import glob
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast

import numpy as np
from PIL import Image

# Try import model classes from repo - adapt if names differ
try:
    from src.models.video_model import DMC as VideoModel
except Exception as e:
    # compress failed; warn and fallback to a differentiable decoder path that uses model parameters
    print("Warning: model.compress() failed during training wrapper:", e)
    try:
        # Do NOT use torch.no_grad() here — we need gradients
        import torch.nn.functional as F
        x = seq[:,0]  # first frame in sequence
        # fallback path: try to run encoder/decoder directly
        q_enc = model.q_encoder[0:1] if hasattr(model,"q_encoder") else None
        q_dec = model.q_decoder[0:1] if hasattr(model,"q_decoder") else None
        q_recon = model.q_recon[0:1] if hasattr(model,"q_recon") else None
        # feature adaptor
        feat_in = F.pixel_unshuffle(x,8)
        if hasattr(model,"feature_adaptor_i"):
            feature = model.feature_adaptor_i(feat_in)
        else:
            feature = feat_in
        ctx, ctx_t = (None,None)
        if hasattr(model,"feature_extractor"):
            qf = model.q_feature[0:1] if hasattr(model,"q_feature") else None
            try:
                ctx, ctx_t = model.feature_extractor(feature, qf)
            except Exception:
                pass
        # encoder -> y
        if hasattr(model,"encoder"):
            y = model.encoder(x, ctx if ctx is not None else feature, q_enc)
        else:
            y = feature
        # decoder -> recon
        if hasattr(model,"decoder") and hasattr(model,"recon_generation_net"):
            feature_dec = model.decoder(y, ctx if ctx is not None else feature, q_dec)
            x_hat = model.recon_generation_net(feature_dec, q_recon).clamp_(0,1)
        else:
            x_hat = x
    except Exception as e2:
        print("Warning: differentiable fallback also failed:", e2)
        x_hat = x
    bits = 0
    x_hat_list = []
    for t in range(T):
        x = seq[:, t]  # [B,C,H,W]

        # compress -> returns dict with 'bit_stream' or other keys
        try:
            out = model.compress(x, qp)
        except Exception as e:
            # compress failed; warn and fallback to identity recon
            print("Warning: model.compress() failed during training wrapper:", e)
            # fallback: run model's internal decoder if available, else use input as reconstruction
            try:
                # try to call decoder modules directly (best-effort)
                with torch.no_grad():
                    # Call encoder+decoder path when available (best-effort)
                    feature = model.apply_feature_adaptor() if hasattr(model, "apply_feature_adaptor") else None
                    ctx, ctx_t = model.feature_extractor(feature, model.q_feature[0:1]) if hasattr(model, "feature_extractor") else (None, None)
                    y = model.encoder(x, ctx, model.q_encoder[0:1]) if hasattr(model, "encoder") else None
                    # try to run hyper/hyper_decoder and decoder path to get y_hat
                    y_hat = y
                    if hasattr(model, "decoder") and y_hat is not None:
                        # try best-effort decode
                        x_hat, _feat = model.get_recon_and_feature(y_hat, ctx, model.q_decoder[0:1], model.q_recon[0:1])
                    else:
                        x_hat = x
                bits = 0
            except Exception:
                x_hat = x
                bits = 0
            x_hat_list.append(x_hat.unsqueeze(1))
            total_bits += bits
            continue

        # get bit stream bytes (may be None)
        bs = out.get("bit_stream", None)
        # if compress returned x_hat directly (some image models do), pick it
        x_hat = out.get("x_hat", None)
        if bs is None:
            # No bitstream; maybe single-frame image model returned x_hat; handle it
            if x_hat is None:
                # fallback: use identity
                x_hat = x
                bits = 0
            else:
                bits = 0
        else:
            # bs may be bytes or bytearray
            if isinstance(bs, (bytes, bytearray)):
                bits = len(bs) * 8
            elif hasattr(bs, "__len__"):
                try:
                    bits = len(bs) * 8
                except Exception:
                    bits = 0
            else:
                bits = 0

            # decompress to get reconstruction
            try:
                sps = {'height': H, 'width': W, 'ec_part': 0}
                dec = model.decompress(bs, sps, qp)
                x_hat = dec.get('x_hat', x)
            except Exception as e:
                # decompress failed, fallback to input
                print("Warning: model.decompress() failed inside training wrapper:", e)
                x_hat = x

        x_hat_list.append(x_hat.unsqueeze(1))
        total_bits += bits

    x_hat_seq = torch.cat(x_hat_list, dim=1) if len(x_hat_list) > 0 else seq
    # normalize bpp per-frame with same denom used elsewhere in training.py (num_pixels = B*H*W)
    denom = float(B * H * W) if (B * H * W) > 0 else 1.0
    bpp_val = float(total_bits) / denom
    # return bpp as a scalar tensor on device
    bpp_tensor = torch.tensor(bpp_val, device=device, dtype=torch.float32)
    return {'x_hat': x_hat_seq, 'likelihoods': None, 'bpp_tensor': bpp_tensor}



def run_epoch(model, loader, optimizer, scaler, device, args, epoch, is_train=True):
    model.train() if is_train else model.eval()
    total_loss = 0.0
    total_dist = 0.0
    total_bpp = 0.0
    steps = 0
    startt = time.time()

    use_video_api = hasattr(model, "compress") and hasattr(model, "decompress")

    for it, seq in enumerate(loader):
        # seq: [B,T,3,H,W]
        seq = seq.to(device, non_blocking=True)
        B,T,C,H,W = seq.shape
        num_pixels = B * H * W  # normalized per-frame; keep same denom as previous code

        # run model: if video API available use encode/decode wrapper, else call model(seq)
        if use_video_api:
            out = encode_decode_sequence(model, seq, qp=0)
            x_hat = out.get('x_hat')
            likelihoods = out.get('likelihoods', None)
            # bpp tensor provided directly by wrapper
            bpp = out.get('bpp_tensor', torch.tensor(0.0, device=device))
            # ensure shapes align: x_hat [B,T,C,H,W]
        else:
            try:
                with autocast(enabled=args.amp):
                    out = model(seq)
            except Exception:
                # fallback: try to run per-frame image model
                flat = seq.view(B*T, C, H, W)
                with autocast(enabled=args.amp):
                    out_flat = model(flat)
                if isinstance(out_flat, dict):
                    x_hat_flat = out_flat.get('x_hat') or out_flat.get('recon') or out_flat.get('x_rec')
                    likelihoods = out_flat.get('likelihoods') or out_flat.get('y_likelihoods') or out_flat.get('lik')
                elif isinstance(out_flat, (list, tuple)):
                    x_hat_flat = out_flat[0]
                    likelihoods = out_flat[1] if len(out_flat) > 1 else None
                else:
                    x_hat_flat = None
                if x_hat_flat is not None and x_hat_flat.dim() == 4 and x_hat_flat.shape[0] == B*T:
                    x_hat = x_hat_flat.view(B, T, C, H, W)
                else:
                    x_hat = seq.clone()
                bpp = torch.tensor(0.0, device=device)

        # if not set yet (video wrapper set it), and out is dict from model
        if not use_video_api:
            if isinstance(out, dict):
                x_hat = out.get('x_hat') or out.get('recon') or out.get('x_rec')
                likelihoods = out.get('likelihoods') or out.get('y_likelihoods') or out.get('lik')
                if x_hat is not None and x_hat.dim() == 4 and x_hat.shape[0] == B*T:
                    x_hat = x_hat.view(B, T, C, H, W)
            elif isinstance(out, (list, tuple)):
                x_hat = out[0]
            else:
                # ensure we have x_hat
                x_hat = seq.clone()

            if 'bpp_tensor' in out:
                bpp = out['bpp_tensor']
            else:
                # compute bpp from likelihoods if available
                bpp = compute_bpp_from_likelihoods(likelihoods, num_pixels).to(device)

        # MSE over all frames and channels
        dist = nn.functional.mse_loss(x_hat, seq, reduction='mean')
        # if bpp is a tensor scalar or a python float
        if isinstance(bpp, torch.Tensor):
            bpp_val = bpp
        else:
            bpp_val = torch.tensor(float(bpp), device=device)

        loss = dist + args.lambda_rd * bpp_val

        if is_train:
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            if args.max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
            scaler.step(optimizer)
            scaler.update()

        total_loss += float(loss.detach().cpu().item())
        total_dist += float(dist.detach().cpu().item())
        total_bpp += float(bpp_val.detach().cpu().item())
        steps += 1

        if is_train and it % args.log_interval == 0:
            print(f"{'Train' if is_train else 'Val'} Epoch {epoch} it {it}/{len(loader)} loss={loss.item():.6f} dist={dist.item():.6f} bpp={bpp_val.item():.6f}")

    elapsed = time.time() - startt
    avg_loss = total_loss / max(1, steps)
    avg_dist = total_dist / max(1, steps)
    avg_bpp = total_bpp / max(1, steps)
    print(f"{'Train' if is_train else 'Val'} Epoch {epoch} finished: avg_loss={avg_loss:.6f} avg_dist={avg_dist:.6f} avg_bpp={avg_bpp:.6f} throughput_steps_per_sec={steps/max(1e-6,elapsed):.2f}")
    return avg_loss, avg_dist, avg_bpp

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-glob', type=str, required=True)
    parser.add_argument('--val-glob', type=str, default='')
    parser.add_argument('--patch-size', type=int, default=256)
    parser.add_argument('--temporal-len', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=6)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-6)
    parser.add_argument('--lambda-rd', type=float, default=0.01)
    parser.add_argument('--pretrained', type=str, default='')
    parser.add_argument('--save-dir', type=str, default='./checkpoints_finetune')
    parser.add_argument('--log-dir', type=str, default='./logs')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--max-norm', type=float, default=1.0)
    parser.add_argument('--log-interval', type=int, default=50)
    args = parser.parse_args()

    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    print("Device:", device)

    # load file lists
    train_files = sorted(glob(args.train_glob, recursive=True))
    val_files = sorted(glob(args.val_glob, recursive=True)) if args.val_glob else []
    print("Train frames:", len(train_files), "Val frames:", len(val_files))

    train_ds = SequencePatchDataset(train_files, seq_len=args.temporal_len, patch_size=args.patch_size, augment=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)

    val_loader = None
    if len(val_files) > 0:
        val_ds = SequencePatchDataset(val_files, seq_len=args.temporal_len, patch_size=args.patch_size, augment=False)
        val_loader = DataLoader(val_ds, batch_size=max(1, args.batch_size//2), shuffle=False, num_workers=max(1,args.workers//2), pin_memory=True)

    # instantiate model
    if VideoModel is None:
        raise RuntimeError("VideoModel import failed. Edit training.py to import the correct class from src.models.")
    model = VideoModel()
    model.to(device)

    # load pretrained (optional)
    if args.pretrained and os.path.exists(args.pretrained):
        ck = torch.load(args.pretrained, map_location='cpu')
        sd = ck.get('state_dict', ck)
        try:
            model.load_state_dict(sd, strict=False)
            print("Loaded pretrained checkpoint:", args.pretrained)
        except Exception as e:
            print("Warning: load_state_dict strict failed:", e)
            # try to load partial keys
            model.load_state_dict({k.replace('module.',''):v for k,v in sd.items() if isinstance(v, torch.Tensor)}, strict=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler(enabled=args.amp)

    best_val = 1e9
    os.makedirs(args.save_dir, exist_ok=True)

    for epoch in range(args.epochs):
        train_loss, train_dist, train_bpp = run_epoch(model, train_loader, optimizer, scaler, device, args, epoch, is_train=True)
        if val_loader is not None:
            val_loss, val_dist, val_bpp = run_epoch(model, val_loader, optimizer, scaler, device, args, epoch, is_train=False)
            # save best
            if val_loss < best_val:
                best_val = val_loss
                fn = os.path.join(args.save_dir, f"best_epoch{epoch:03d}.pth.tar")
                torch.save({'epoch':epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, fn)
                print("Saved best checkpoint:", fn)

        # periodic save
        if (epoch + 1) % 5 == 0:
            fn = os.path.join(args.save_dir, f"epoch{epoch:03d}.pth.tar")
            torch.save({'epoch':epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, fn)
            print("Saved checkpoint:", fn)

    print("Training finished.")

if __name__ == "__main__":
    main()
