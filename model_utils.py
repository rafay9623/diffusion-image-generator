import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from tqdm.auto import tqdm

# ── Scheduler Helpers ────────────────────────────────────────────────────────

def cosine_beta_schedule(T, s=0.008):
    steps = T + 1
    x = torch.linspace(0, T, steps)
    ac = torch.cos(((x / T) + s) / (1 + s) * math.pi / 2) ** 2
    ac = ac / ac[0]
    betas = 1 - (ac[1:] / ac[:-1])
    return betas.clamp(1e-5, 0.9999)

def linear_beta_schedule(T, b_start=1e-4, b_end=0.02):
    return torch.linspace(b_start, b_end, T)

class DiffusionScheduler:
    def __init__(self, T=300, schedule='cosine'):
        self.T = T
        betas = cosine_beta_schedule(T) if schedule=='cosine' else linear_beta_schedule(T)
        self.betas    = betas
        alphas        = 1.0 - betas
        self.ac       = torch.cumprod(alphas, 0)            # ᾱ_t
        self.ac_prev  = F.pad(self.ac[:-1], (1,0), value=1.)
        self.sqrt_ac      = self.ac.sqrt()
        self.sqrt_1m_ac   = (1 - self.ac).sqrt()
        self.sqrt_recip_a = (1./alphas).sqrt()
        self.post_var     = betas * (1 - self.ac_prev) / (1 - self.ac)

    def _get(self, a, t, shape):
        v = a.gather(-1, t.cpu()).to(t.device)
        return v.reshape(t.shape[0], *([1]*(len(shape)-1)))

    def q_sample(self, x0, t, noise=None):
        """Forward: x_t = sqrt(ᾱ_t)*x0 + sqrt(1-ᾱ_t)*ε"""
        if noise is None: noise = torch.randn_like(x0)
        return self._get(self.sqrt_ac, t, x0.shape)*x0 + \
               self._get(self.sqrt_1m_ac, t, x0.shape)*noise

    def p_losses(self, model, x0, t, noise=None):
        """MSE loss: ||ε - ε_θ(x_t, t)||²"""
        if noise is None: noise = torch.randn_like(x0)
        x_noisy = self.q_sample(x0, t, noise)
        pred    = model(x_noisy, t)
        return F.mse_loss(pred, noise)

    @torch.no_grad()
    def p_sample(self, model, x, t_idx):
        t  = torch.full((x.shape[0],), t_idx, device=x.device, dtype=torch.long)
        bt = self._get(self.betas, t, x.shape)
        s1 = self._get(self.sqrt_1m_ac, t, x.shape)
        sr = self._get(self.sqrt_recip_a, t, x.shape)
        mean = sr * (x - bt * model(x, t) / s1)
        if t_idx == 0: return mean
        pv   = self._get(self.post_var, t, x.shape)
        return mean + pv.sqrt() * torch.randn_like(x)

    @torch.no_grad()
    def sample(self, model, shape, return_all=False):
        dev = next(model.parameters()).device
        img = torch.randn(shape, device=dev)
        imgs = [img.cpu()]
        for i in tqdm(reversed(range(self.T)), desc='Sampling', total=self.T, leave=False):
            img = self.p_sample(model, img, i)
            if return_all and i % (self.T//10) == 0:
                imgs.append(img.cpu())
        return (img.cpu(), imgs) if return_all else img.cpu()

# ── Model Architecture ────────────────────────────────────────────────────────

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, t):
        half = self.dim // 2
        emb  = math.log(10000) / (half - 1)
        emb  = torch.exp(torch.arange(half, device=t.device) * -emb)
        emb  = t.float().unsqueeze(1) * emb.unsqueeze(0)
        return torch.cat([emb.sin(), emb.cos()], dim=-1)

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim * 4),
        )
    def forward(self, t): return self.net(t)

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, groups=8):
        super().__init__()
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_ch))
        self.block1   = nn.Sequential(nn.GroupNorm(groups, in_ch),  nn.SiLU(), nn.Conv2d(in_ch,  out_ch, 3, padding=1))
        self.block2   = nn.Sequential(nn.GroupNorm(groups, out_ch), nn.SiLU(), nn.Conv2d(out_ch, out_ch, 3, padding=1))
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.block1(x)
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        h = self.block2(h)
        return h + self.shortcut(x)

class AttentionBlock(nn.Module):
    def __init__(self, ch, heads=4):
        super().__init__()
        self.norm  = nn.GroupNorm(8, ch)
        self.attn  = nn.MultiheadAttention(ch, heads, batch_first=True)
    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x).reshape(B, C, H*W).transpose(1,2)
        h, _ = self.attn(h, h, h)
        return x + h.transpose(1,2).reshape(B, C, H, W)

class UNet(nn.Module):
    def __init__(self, in_ch=3, base_dim=64, dim_mults=(1,2,4),
                 attn_res=16, image_size=128):
        super().__init__()
        time_emb_dim = base_dim * 4
        dims   = [base_dim * m for m in dim_mults]
        in_out = list(zip([base_dim]+dims[:-1], dims))

        self.time_emb = TimeEmbedding(base_dim)
        self.init_conv = nn.Conv2d(in_ch, base_dim, 7, padding=3)

        self.downs = nn.ModuleList()
        self.down_sample = nn.ModuleList()
        for i, (d_in, d_out) in enumerate(in_out):
            self.downs.append(nn.ModuleList([
                ResBlock(d_in,  d_out, base_dim*4),
                ResBlock(d_out, d_out, base_dim*4),
                AttentionBlock(d_out) if (image_size // (2**i)) <= attn_res else nn.Identity(),
            ]))
            self.down_sample.append(
                nn.Conv2d(d_out, d_out, 4, 2, 1) if i < len(in_out)-1 else nn.Identity()
            )

        mid = dims[-1]
        self.mid_res1 = ResBlock(mid, mid, base_dim*4)
        self.mid_attn = AttentionBlock(mid)
        self.mid_res2 = ResBlock(mid, mid, base_dim*4)

        self.ups = nn.ModuleList()
        self.up_sample = nn.ModuleList()
        for i, (d_in, d_out) in enumerate(reversed(in_out)):
            self.ups.append(nn.ModuleList([
                ResBlock(d_out*2, d_in, base_dim*4),
                ResBlock(d_in,   d_in, base_dim*4),
                AttentionBlock(d_in) if (image_size // (2**(len(in_out)-i-1))) <= attn_res else nn.Identity(),
            ]))
            self.up_sample.append(
                nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                              nn.Conv2d(d_in, d_in, 3, padding=1))
                if i < len(in_out)-1 else nn.Identity()
            )

        self.final = nn.Sequential(
            nn.GroupNorm(8, base_dim),
            nn.SiLU(),
            nn.Conv2d(base_dim, in_ch, 1),
        )

    def forward(self, x, t):
        t_emb = self.time_emb(t)
        x = self.init_conv(x)
        skips = []

        for (r1, r2, attn), ds in zip(self.downs, self.down_sample):
            x = r1(x, t_emb); x = r2(x, t_emb); x = attn(x)
            skips.append(x)
            x = ds(x)

        x = self.mid_res1(x, t_emb)
        x = self.mid_attn(x)
        x = self.mid_res2(x, t_emb)

        for (r1, r2, attn), us in zip(self.ups, self.up_sample):
            x = torch.cat([x, skips.pop()], dim=1)
            x = r1(x, t_emb); x = r2(x, t_emb); x = attn(x)
            x = us(x)

        return self.final(x)
