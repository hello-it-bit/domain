import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.attention import Attention


# ============================================================
# 1) FEI: Frequency-Energy Indicator (DoG bands + normalized energy)
#    Input:  z_t  [B, C, H, W]
#    Output: e_t  [B, num_bands]
# ============================================================
# class FrequencyEnergyIndicator(nn.Module):
#     def __init__(self, num_bands: int = 3, eps: float = 1e-8):
#         super().__init__()
#         assert num_bands >= 2, "num_bands should be >= 2"
#         self.num_bands = num_bands
#         self.eps = eps

#         # Simple sigma schedule (can be tuned)
#         # num_bands => num_scales = num_bands + 1 for DoG
#         # e.g. 3 bands => 4 scales
#         base = 0.6
#         ratio = 1.6
#         sigmas = [base * (ratio ** i) for i in range(num_bands + 1)]
#         self.register_buffer("sigmas", torch.tensor(sigmas, dtype=torch.float32), persistent=False)

#     @staticmethod
#     def _gaussian_blur2d(x: torch.Tensor, sigma: float) -> torch.Tensor:
#         """
#         Lightweight gaussian blur using separable conv.
#         x: [B,C,H,W]
#         """
#         if sigma <= 0:
#             return x

#         # kernel size ~ 6*sigma, make odd and at least 3
#         k = int(max(3, (sigma * 6 + 1) // 2 * 2 + 1))
#         device, dtype = x.device, x.dtype

#         # 1D gaussian
#         coords = torch.arange(k, device=device, dtype=dtype) - (k - 1) / 2
#         g = torch.exp(-(coords ** 2) / (2 * (sigma ** 2)))
#         g = g / (g.sum() + 1e-12)

#         # separable conv
#         g1 = g.view(1, 1, 1, k)
#         g2 = g.view(1, 1, k, 1)

#         # depthwise
#         B, C, H, W = x.shape
#         x_ = x.view(B * C, 1, H, W)
#         x_ = F.conv2d(x_, g1, padding=(0, k // 2))
#         x_ = F.conv2d(x_, g2, padding=(k // 2, 0))
#         return x_.view(B, C, H, W)

#     def forward(self, z_t: torch.Tensor) -> torch.Tensor:
#         """
#         Returns normalized band energies e_t: [B, num_bands]
#         """
#         assert z_t.dim() == 4, f"FEI expects [B,C,H,W], got {tuple(z_t.shape)}"

#         # build gaussian pyramid
#         gaussians = []
#         for s in self.sigmas.tolist():
#             gaussians.append(self._gaussian_blur2d(z_t, float(s)))

#         # DoG bands
#         band_components = []
#         for i in range(self.num_bands):
#             band = gaussians[i] - gaussians[i + 1]
#             band_components.append(band)

#         # energy per band
#         # energy: sum over C,H,W
#         energies = []
#         for band in band_components:
#             # e = (band ** 2).mean(dim=(1, 2, 3))  # [B]
#             e = (band ** 2).sum(dim=(1, 2, 3))
#             energies.append(e)
#         energies = torch.stack(energies, dim=1)  # [B, num_bands]

#         # normalize to distribution
#         e_sum = energies.sum(dim=1, keepdim=True) + self.eps
#         e_t = energies / e_sum
#         return e_t

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FrequencyEnergyIndicator(nn.Module):
    """
    FeRA-style FEI with explicit low/mid/high semantics:
      - low  : G(sigma_last) * z
      - mid  : G(sigma_i) * z - G(sigma_{i+1}) * z
      - high : z - G(sigma_0) * z
    Returns:
      e_t: (B, num_bands) normalized band energies
      band_components: List[Tensor], each (B, C, H, W), ordered [low, ..., high]
    """
    def __init__(self, num_bands: int = 3):
        super().__init__()
        assert num_bands >= 2, "num_bands must be >= 2"
        self.num_bands = num_bands
        # For num_bands=3 -> scales = [1, 2] -> sigmas = [kappa*1, kappa*2]
        self.scales = [2 ** k for k in range(num_bands - 1)]

    @staticmethod
    def _gaussian_kernel2d(kernel_size: int, sigma: float, channels: int, device, dtype):
        # kernel_size must be odd
        x = torch.arange(kernel_size, device=device, dtype=dtype) - (kernel_size - 1) / 2.0
        x_grid = x.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy = torch.stack([x_grid, y_grid], dim=-1)

        var = sigma * sigma
        kernel = (1.0 / (2.0 * math.pi * var)) * torch.exp(-torch.sum(xy * xy, dim=-1) / (2.0 * var))
        kernel = kernel / kernel.sum()

        # depthwise conv2d weights: (C, 1, k, k)
        return kernel.view(1, 1, kernel_size, kernel_size).repeat(channels, 1, 1, 1)

    def _blur(self, x: torch.Tensor, sigma: float) -> torch.Tensor:
        if sigma <= 0:
            return x
        # Same heuristic as FeRA: k ~ 8*sigma + 1 (force odd)
        k = int(2 * 4 * sigma + 1)
        k = k | 1  # make odd
        pad = k // 2

        B, C, H, W = x.shape
        weight = self._gaussian_kernel2d(k, sigma, C, x.device, x.dtype)
        return F.conv2d(x, weight, padding=pad, groups=C)

    def forward(self, z_t: torch.Tensor):
        """
        z_t: (B, C, H, W)
        """
        assert z_t.ndim == 4, f"Expected (B,C,H,W), got {z_t.shape}"
        B, C, H, W = z_t.shape

        # FeRA uses kappa = min(H, W)/128
        kappa = min(H, W) / 128.0
        sigmas = [kappa * s for s in self.scales]  # length = num_bands-1

        # Blurs: b0 = G(sigma0)*z, b1 = G(sigma1)*z, ...
        blurs = [self._blur(z_t, s) for s in sigmas]

        # Construct bands in FeRA order: [low, mid..., high]
        # low
        band_components = [blurs[-1]]
        # mids (descending frequency: sigma_{i} - sigma_{i+1})
        for i in range(len(blurs) - 2, -1, -1):
            band_components.append(blurs[i] - blurs[i + 1])
        # high
        band_components.append(z_t - blurs[0])

        # If num_bands > 3, above yields exactly num_bands bands:
        # 1(low) + (num_bands-2)(mid) + 1(high) = num_bands
        # For safety, trim/pad if any mismatch (shouldn't happen)
        if len(band_components) > self.num_bands:
            band_components = band_components[: self.num_bands]
        elif len(band_components) < self.num_bands:
            # pad with zeros (rare)
            pad_n = self.num_bands - len(band_components)
            band_components += [torch.zeros_like(z_t) for _ in range(pad_n)]

        # Energy per band: sum over C,H,W
        energies = [torch.sum(b * b, dim=(1, 2, 3)) for b in band_components]  # list of (B,)
        energy_vec = torch.stack(energies, dim=1)  # (B, num_bands)

        e_t = energy_vec / (energy_vec.sum(dim=1, keepdim=True) + 1e-8)
        return e_t, band_components



# ============================================================
# 2) Router: small MLP -> alpha [B, num_experts]
# ============================================================
class SoftFrequencyRouter(nn.Module):
    def __init__(self, num_bands: int, num_experts: int, tau: float = 0.7, hidden: int = 64):
        super().__init__()
        self.num_bands = num_bands
        self.num_experts = num_experts
        self.tau = tau

        self.mlp = nn.Sequential(
            nn.Linear(num_bands, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, num_experts),
        )

    def forward(self, e_t: torch.Tensor) -> torch.Tensor:
        # e_t: [B, num_bands]
        logits = self.mlp(e_t)
        alpha = F.softmax(logits / max(self.tau, 1e-6), dim=-1)
        return alpha


# ============================================================
# 3) Shared routing state: one FEI+Router shared across all processors
#    You call update_from_latent() once per step in train.py
# ============================================================
class FeRARoutingState(nn.Module):
    def __init__(
        self,
        num_bands: int = 3,
        num_experts: int = 3,
        router_tau: float = 0.7,
        router_hidden: int = 64,
        detach_alpha: bool = False,
    ):
        super().__init__()
        self.fei = FrequencyEnergyIndicator(num_bands=num_bands)
        self.router = SoftFrequencyRouter(num_bands=num_bands, num_experts=num_experts, tau=router_tau, hidden=router_hidden)
        self.detach_alpha = detach_alpha

        self.alpha: Optional[torch.Tensor] = None  # [B, K]

  
    def clear(self):
        self.alpha = None
        self.e_t = None

    def update_from_latent(self, latent2d: torch.Tensor) -> torch.Tensor:
        """
        latent2d: [B,C,H,W] (建议用 noisy video latent 做 time mean 得到 2D)
        returns alpha: [B,K]
        """
        # e_t = self.fei(latent2d)                 # [B, num_bands]
        e_t, _ = self.fei(latent2d)
        alpha = self.router(e_t)                 # [B, num_experts]
        if self.detach_alpha:
            alpha = alpha.detach()
        self.alpha = alpha
        self.e_t = e_t
        return alpha


# ============================================================
# 4) Multi-expert LoRA linear layer (keeps total rank constant)
# ============================================================
class MultiExpertLoRALinearLayer(nn.Module):
    """
    Implements: sum_k alpha_k * LoRA_k(x)
    where each LoRA_k is a low-rank (down->up) without bias.
    """
    def __init__(self, in_features: int, out_features: int, total_rank: int, num_experts: int = 3, network_alpha=None,
                 device=None, dtype=None):
        super().__init__()
        assert num_experts >= 1
        self.in_features = in_features
        self.out_features = out_features
        self.num_experts = num_experts
        self.network_alpha = network_alpha
        self.total_rank = total_rank

        # keep total parameter roughly constant:
        # total_rank = sum(rank_k). use uniform split.
        
        assert total_rank >= num_experts,\
            f"total_rank ({total_rank}) must >= num_experts ({num_experts}) to keep params constant"
        # rank_per = total_rank // num_experts
        # # if total_rank < num_experts, some experts will still have rank=1 (slightly more params than original)
        # self.rank_per_expert = rank_per

        base = total_rank // num_experts
        rem  = total_rank % num_experts
        ranks = [base + (1 if k < rem else 0) for k in range(num_experts)]
        assert sum(ranks) == total_rank
        self.rank_per_expert = ranks

        # self.down = nn.ModuleList([
        #     nn.Linear(in_features, rank_per, bias=False, device=device, dtype=dtype)
        #     for _ in range(num_experts)
        # ])
        # self.up = nn.ModuleList([
        #     nn.Linear(rank_per, out_features, bias=False, device=device, dtype=dtype)
        #     for _ in range(num_experts)
        # ])

        self.down = nn.ModuleList([
            nn.Linear(in_features, ranks[k], bias=False, device=device, dtype=dtype)
            for k in range(num_experts)
        ])
        self.up = nn.ModuleList([
            nn.Linear(ranks[k], out_features, bias=False, device=device, dtype=dtype)
            for k in range(num_experts)
        ])


        # init like standard LoRA
        for k in range(num_experts):
            # nn.init.normal_(self.down[k].weight, std=1 / max(rank_per, 1))
            nn.init.normal_(self.down[k].weight, std=1 / max(self.rank_per_expert[k], 1))
            nn.init.zeros_(self.up[k].weight)

    def forward(self, hidden_states: torch.Tensor, alpha: Optional[torch.Tensor]) -> torch.Tensor:
        """
        hidden_states: [B, N, D]
        alpha: [B, K] or None
        returns delta: [B, N, out_features]
        """
        if alpha is None:
            # no routing => no adapter contribution
            # return hidden_states.new_zeros(hidden_states.shape[0], hidden_states.shape[1], self.out_features)
            return torch.zeros(hidden_states.shape[0],hidden_states.shape[1],self.out_features,device=hidden_states.device,dtype=hidden_states.dtype,)

        B, N, _ = hidden_states.shape
        orig_dtype = hidden_states.dtype
        dtype = self.down[0].weight.dtype

        # compute each expert output
        deltas = []
        hs = hidden_states.to(dtype)
        for k in range(self.num_experts):
            d = self.up[k](self.down[k](hs))  # [B,N,out]
            if self.network_alpha is not None:
                # d = d * (self.network_alpha / max(self.rank_per_expert, 1))
                d = d * (self.network_alpha / max(self.rank_per_expert[k], 1))

            deltas.append(d)
        # [B,K,N,out]
        deltas = torch.stack(deltas, dim=1)

        # weight and sum
        # alpha: [B,K] -> [B,K,1,1]
        w = alpha[:, :, None, None].to(deltas.dtype)
        out = (w * deltas).sum(dim=1)  # [B,N,out]
        return out.to(orig_dtype)


# ============================================================
#    FeRA-RefAdapter Processor
#    - stage=None: RefAdapter + FeRA routing (multi-expert LoRA)
#    - stage!=None: keep original FAE token logic (no FeRA needed)
# ============================================================
class FeRARefNetLoRAProcessor(nn.Module):
    """
    Drop-in replacement for RefNetLoRAProcessor.
    You MUST pass a shared FeRARoutingState for stage=None (RefAdapter).
    """

    def __init__(
        self,
        dim: int,
        rank: int = 4,
        network_alpha=None,
        lora_weight: float = 1.0,
        stage=None,

        # motion inversion tokens (FAE stage)
        num_motion_tokens: int = 1024,
        is_train: bool = True,

        # attn reweight
        reweight_scale=None,
        vid2embed: int = 1,
        embed2vid: int = 0,

        # visualize
        attn_map_save_path: Optional[str] = None,
        cur_layer: int = 0,
        cur_step: int = 0,
        save_step=(10,),
        save_layer=(40,),

        # ===== FeRA params (RefAdapter stage) =====
        routing_state: Optional[FeRARoutingState] = None,
        num_experts: int = 3,        # FeRA paper常用 3 experts
    ):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("CogVideoXAttnProcessor requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

        self.lora_weight = float(lora_weight)
        self.stage = stage
        self.is_train = is_train
        self.attention_mask = None

        # Attn reweight
        self.reweight_scale = reweight_scale
        self.vid2embed = vid2embed
        self.embed2vid = embed2vid

        # Visualization
        self.attn_map_save_path = attn_map_save_path
        self.cur_layer = cur_layer
        self.cur_step = cur_step
        self.save_step = list(save_step)
        self.save_layer = list(save_layer)

        # ===== FeRA routing shared state (only used in RefAdapter stage) =====
        self.routing_state = routing_state
        self.num_experts = num_experts

        # Multi-expert LoRA (RefAdapter stage)
        # Keep total rank constant by splitting it into experts internally.
        self.lora_q = MultiExpertLoRALinearLayer(dim, dim, total_rank=rank, num_experts=num_experts, network_alpha=network_alpha)
        self.lora_k = MultiExpertLoRALinearLayer(dim, dim, total_rank=rank, num_experts=num_experts, network_alpha=network_alpha)
        self.lora_v = MultiExpertLoRALinearLayer(dim, dim, total_rank=rank, num_experts=num_experts, network_alpha=network_alpha)
        self.lora_proj = MultiExpertLoRALinearLayer(dim, dim, total_rank=rank, num_experts=num_experts, network_alpha=network_alpha)

        # Stage 2 (FAE)
        if stage is not None:
            self.num_motion_tokens = num_motion_tokens
            self.motion_inversion_tokens = nn.Parameter(torch.zeros(1, num_motion_tokens, dim))
            nn.init.zeros_(self.motion_inversion_tokens)
            print(
                f"[FAE-INIT] motion_inversion_tokens shape={self.motion_inversion_tokens.shape}, "
                f"num_tokens={self.num_motion_tokens}, emb_dim={dim}, total_params={self.motion_inversion_tokens.numel()}"
            )

    def save_attn_map(self, q_vis, k_vis, v_vis, save_step=(10,), save_layer=(40,)):
        attn_map = torch.zeros_like(v_vis)
        if not os.path.exists(os.path.join(self.attn_map_save_path, "status.txt")):
            with open(os.path.join(self.attn_map_save_path, "status.txt"), "w") as f:
                f.write("0")
                cur_status = 0
        else:
            with open(os.path.join(self.attn_map_save_path, "status.txt"), "r") as f:
                cur_status = int(f.read()) + 1
            with open(os.path.join(self.attn_map_save_path, "status.txt"), "w") as f:
                f.write(f"{cur_status}")

        cur_step = cur_status // 42
        cur_layer = cur_status % 42

        if cur_step in save_step and cur_layer in save_layer:
            print(f"save attn map at step {cur_step} layer {cur_layer}")
            for i in range(q_vis.shape[1]):
                q_mini = q_vis[:, i : i + 1]
                k_mini = k_vis[:, i : i + 1]
                attn_map_mini = F.scaled_dot_product_attention(
                    q_mini, k_mini, v_vis, attn_mask=None, dropout_p=0.0, is_causal=False
                )
                attn_map += attn_map_mini
            save_path = os.path.join(self.attn_map_save_path, f"{cur_step}_{cur_layer}.pt")
            torch.save(attn_map, save_path)

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ):
        text_seq_length = encoder_hidden_states.size(1)

        # ---- stage switch ----
        if self.stage is not None:
            # FAE stage: keep original behavior
            cat_tokens = self.motion_inversion_tokens.repeat(encoder_hidden_states.size(0), 1, 1)
            encoder_hidden_states = torch.cat([encoder_hidden_states, cat_tokens], dim=1)
            encoder_h_seq_length = encoder_hidden_states.size(1)
            hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        else:
            # RefAdapter stage
            encoder_h_seq_length = encoder_hidden_states.size(1)
            hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        # ===== FeRA alpha: only for RefAdapter stage =====
        alpha = None
        if self.stage is None:
            # shared routing_state is expected
            if self.routing_state is not None:
                alpha = self.routing_state.alpha  # [B,K] or None

        # Q
        query = attn.to_q(hidden_states)
        query = query + self.lora_q(hidden_states, alpha) * self.lora_weight

        # K
        key = attn.to_k(hidden_states)
        key = key + self.lora_k(hidden_states, alpha) * self.lora_weight

        # V
        value = attn.to_v(hidden_states)
        value = value + self.lora_v(hidden_states, alpha) * self.lora_weight

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # RoPE if needed
        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb

            query[:, :, encoder_h_seq_length:] = apply_rotary_emb(query[:, :, encoder_h_seq_length:], image_rotary_emb)
            if not attn.is_cross_attention:
                key[:, :, encoder_h_seq_length:] = apply_rotary_emb(key[:, :, encoder_h_seq_length:], image_rotary_emb)

        # attn reweight
        if self.reweight_scale is not None:
            q_len = query.shape[-2]
            attention_mask = torch.zeros((q_len, q_len), device=query.device, dtype=query.dtype)
            if self.vid2embed:
                attention_mask[encoder_h_seq_length:, text_seq_length:encoder_h_seq_length] += self.reweight_scale
            if self.embed2vid:
                attention_mask[text_seq_length:encoder_h_seq_length, encoder_h_seq_length:] += self.reweight_scale

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        if self.attn_map_save_path is not None:
            q_vis = query[1:2].clone().to("cuda")
            k_vis = key[1:2].clone().to("cuda")
            v_vis = torch.eye(q_vis.shape[-2]).to("cuda", dtype=q_vis.dtype).unsqueeze(0).unsqueeze(0)
            q_vis.requires_grad = False
            k_vis.requires_grad = False
            v_vis.requires_grad = False
            self.save_attn_map(q_vis, k_vis, v_vis, save_step=self.save_step, save_layer=self.save_layer)

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states) + self.lora_proj(hidden_states, alpha) * self.lora_weight
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        encoder_hidden_states, hidden_states = hidden_states.split(
            [encoder_h_seq_length, hidden_states.size(1) - encoder_h_seq_length], dim=1
        )
        encoder_hidden_states = encoder_hidden_states[:, :text_seq_length, :]
        return hidden_states, encoder_hidden_states
