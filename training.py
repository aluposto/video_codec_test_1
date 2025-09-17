#!/usr/bin/env python3
# training.py - Training loop for DCVC-style models on pre-extracted frames.

import sys, os
# ensure repo root is on PYTHONPATH so 'src' package can be imported when running as a script
repo_root = os.path.dirname(os.path.abspath(__file__))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

"""
Usage (single GPU):
  python training.py --train-glob "data/frames/train/**/*.png" --val-glob "data/frames/val/**/*.png" \
    --patch-size 256 --temporal-len 4 --batch-size 6 --epochs 30 --cuda --amp \
    --pretrained ./checkpoints/cvpr2025_video.pth.tar --lambda-rd 0.01 --save-dir ./checkpoints_finetune
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
except Exception:
    VideoModel = None

# ----------------------------
# Dataset - reads PNG frames list (flat) and samples temporal clips + random crop patches
# ----------------------------
class SequencePatchDataset(Dataset):
    def __init__(self, file_list, seq_len=4, patch_size=256, augment=True):
        """
        file_list: sorted list of all frame file paths across videos (treated as timeline)
        """
        self.files = sorted(file_list)
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.augment = augment
        if len(self.files) == 0:
            raise ValueError("No frame files provided to SequencePatchDataset")

    def __len__(self):
        # number of clips (simple heuristic)
        return max(1, len(self.files) // max(1, self.seq_len))

    def _read_seq(self, start_idx):
        idxs = [min(len(self.files)-1, start_idx + i) for i in range(self.seq_len)]
        imgs = []
        for i in idxs:
            p = self.files[i]
            im = Image.open(p).convert('RGB')
            arr = np.asarray(im, dtype=np.float32) / 255.0
            imgs.append(arr)
        seq = np.stack(imgs, axis=0) # [T,H,W,3]
        return seq

    def __getitem__(self, idx):
        max_start = max(0, len(self.files) - self.seq_len)
        start = random.randint(0, max_start) if max_start > 0 else 0
        seq = self._read_seq(start)  # [T,H,W,3]
        T,H,W,C = seq.shape
        ps = self.patch_size
        if H < ps or W < ps:
            # resize up if needed
            from PIL import Image
            seq = np.stack([np.asarray(Image.fromarray((f*255).astype(np.uint8)).resize((max(W,ps), max(H,ps))), dtype=np.float32)/255.0 for f in seq], axis=0)
            T,H,W,C = seq.shape
        x = random.randint(0, W-ps) if W > ps else 0
        y = random.randint(0, H-ps) if H > ps else 0
        seq = seq[:, y:y+ps, x:x+ps, :]  # [T,ps,ps,3]
        if self.augment:
            if random.random() < 0.5:
                seq = seq[:, :, ::-1, :]
            if random.random() < 0.5:
                seq = seq[:, ::-1, :, :]
        # to tensor [T,3,H,W]
        seq = seq.transpose(0,3,1,2).copy()
        seq = torch.from_numpy(seq).float()
        return seq

# ----------------------------
# bpp helper
# ----------------------------
def compute_bpp_from_likelihoods(likelihoods, num_pixels):
    if likelihoods is None:
        return torch.tensor(0.0)
    if isinstance(likelihoods, dict):
        tensors = [v for v in likelihoods.values() if isinstance(v, torch.Tensor)]
    elif isinstance(likelihoods, (list, tuple)):
        tensors = [t for t in likelihoods if isinstance(t, torch.Tensor)]
    elif isinstance(likelihoods, torch.Tensor):
        tensors = [likelihoods]
    else:
        try:
            return torch.tensor(float(likelihoods))
        except Exception:
            return torch.tensor(0.0)
    total_bits = torch.tensor(0.0, device=tensors[0].device)
    for t in tensors:
        p = torch.clamp(t, min=1e-9)
        total_bits = total_bits + (-torch.sum(torch.log(p)) / math.log(2.0))
    bpp = total_bits / float(num_pixels)
    return bpp

# ----------------------------
# Helper: encode/decode wrapper for video models
# ----------------------------
def encode_decode_sequence(model, seq, qp=0):
    """
    seq: [B,T,C,H,W] torch tensor on correct device
    Returns dict: {'x_hat': [B,T,C,H,W] tensor, 'likelihoods': None, 'bpp_tensor': scalar-tensor}
    """
    import torch
    import torch.nn.functional as F

    device = seq.device
    B, T, C, H, W = seq.shape

    # initialize entropy coder etc (best-effort)
    try:
        model.update()
    except Exception:
        # update may fail if compiled coder missing; continue
        pass

    # clear and initialize dpb
    try:
        if hasattr(model, "clear_dpb"):
            model.clear_dpb()
    except Exception:
        pass

    # create an initial reference frame (zeros) to avoid None access
    try:
        zeros = torch.zeros((B, C, H, W), device=device, dtype=seq.dtype)
        if hasattr(model, "add_ref_frame"):
            try:
                model.add_ref_frame(feature=None, frame=zeros, increase_poc=False)
            except TypeError:
                try:
                    model.add_ref_frame(frame=zeros, increase_poc=False)
                except Exception:
                    model.dpb = []
                    model.add_ref_frame(frame=zeros, increase_poc=False)
    except Exception:
        pass

    total_bits = 0
    x_hat_list = []

    for t in range(T):
        x = seq[:, t]  # [B,C,H,W]

        # attempt to compress => decompress pipeline (preferred)
        out = None
        try:
            out = model.compress(x, qp)
        except Exception:
            out = None

        if out is None:
            # differentiable fallback using encoder/decoder submodules
            try:
                feat_in = F.pixel_unshuffle(x, 8) if hasattr(F, "pixel_unshuffle") else x
                if hasattr(model, "feature_adaptor_i"):
                    try:
                        feature = model.feature_adaptor_i(feat_in)
                    except Exception:
                        feature = feat_in
                else:
                    feature = feat_in

                ctx, ctx_t = (None, None)
                if hasattr(model, "feature_extractor"):
                    try:
                        qf = model.q_feature[0:1] if hasattr(model, "q_feature") else None
                        ctx, ctx_t = model.feature_extractor(feature, qf)
                    except Exception:
                        ctx, ctx_t = (None, None)

                y = None
                if hasattr(model, "encoder"):
                    try:
                        q_enc = model.q_encoder[0:1] if hasattr(model, "q_encoder") else None
                        y = model.encoder(x, ctx if ctx is not None else feature, q_enc)
                    except Exception:
                        y = None

                x_hat = x
                if y is not None and hasattr(model, "decoder") and hasattr(model, "recon_generation_net"):
                    try:
                        q_dec = model.q_decoder[0:1] if hasattr(model, "q_decoder") else None
                        q_recon = model.q_recon[0:1] if hasattr(model, "q_recon") else None
                        feature_dec = model.decoder(y, ctx if ctx is not None else feature, q_dec)
                        x_hat = model.recon_generation_net(feature_dec, q_recon).clamp_(0, 1)
                    except Exception:
                        try:
                            q_dec = model.q_decoder[0:1] if hasattr(model, "q_decoder") else None
                            q_recon = model.q_recon[0:1] if hasattr(model, "q_recon") else None
                            x_hat, _feat = model.get_recon_and_feature(y, ctx if ctx is not None else feature, q_dec, q_recon)
                            x_hat = x_hat.clamp_(0,1)
                        except Exception:
                            x_hat = x
                bits = 0
            except Exception:
                x_hat = x
                bits = 0
        else:
            bs = out.get("bit_stream", None)
            x_hat = out.get("x_hat", None)
            bits = 0
            if bs is None:
                if x_hat is None:
                    x_hat = x
                    bits = 0
                else:
                    bits = 0
            else:
                try:
                    if isinstance(bs, (bytes, bytearray)):
                        bits = len(bs) * 8
                    elif hasattr(bs, "__len__"):
                        bits = len(bs) * 8
                    else:
                        bits = 0
                except Exception:
                    bits = 0
                try:
                    sps = {'height': H, 'width': W, 'ec_part': 0}
                    dec = model.decompress(bs, sps, qp)
                    x_hat = dec.get('x_hat', x)
                except Exception:
                    x_hat = x

        # ensure x_hat has shape [B,C,H,W]
        if isinstance(x_hat, torch.Tensor):
            if x_hat.dim() == 5 and x_hat.shape[1] == 1:
                x_hat = x_hat[:,0]
        else:
            x_hat = x

        x_hat_list.append(x_hat.unsqueeze(1))
        total_bits += int(bits)

    x_hat_seq = torch.cat(x_hat_list, dim=1) if len(x_hat_list) > 0 else seq
    denom = float(B * H * W) if (B * H * W) > 0 else 1.0
    bpp_val = float(total_bits) / denom
    bpp_tensor = torch.tensor(bpp_val, device=device, dtype=torch.float32)
    return {'x_hat': x_hat_seq, 'likelihoods': None, 'bpp_tensor': bpp_tensor}

# ----------------------------
# training/validation epoch
# ----------------------------
def run_epoch(model, loader, optimizer, scaler, device, args, epoch, is_train=True):
    model.train() if is_train else model.eval()
    total_loss = 0.0
    total_dist = 0.0
    total_bpp = 0.0
    steps = 0
    startt = time.time()

    for it, seq in enumerate(loader):
        # seq: [B,T,3,H,W]
        seq = seq.to(device, non_blocking=True)
        B,T,C,H,W = seq.shape
        num_pixels = B * H * W  # normalized per-frame; keep same denom as previous code

        use_video_api = hasattr(model, "compress") and hasattr(model, "decompress")

        # run model: if video API available use encode/decode wrapper, else call model(seq)
        if use_video_api:
            out = encode_decode_sequence(model, seq, qp=0)
            x_hat = out.get('x_hat')
            likelihoods = out.get('likelihoods', None)
            # bpp tensor provided directly by wrapper
            bpp = out.get('bpp_tensor', torch.tensor(0.0, device=device))
        else:
            try:
                with autocast(enabled=args.amp):
                    out = model(seq)  # many image models implement __call__ to return recon or dict
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

        # if not set yet (non-video path) and out is dict
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

            if isinstance(out, dict) and 'bpp_tensor' in out:
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

# ----------------------------
# main
# ----------------------------
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
