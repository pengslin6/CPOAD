"""CPCD4: intra-physical + intra-cyber predictive consistency.

Same single mechanism as CPCD3 (Normal-only NLL predictive consistency),
applied in two intra-domain directions:
  1. Intra-physical (from CPCD3): predict each physical sensor from others.
     Captures inter-sensor correlation anomalies (e.g. spoofed sensor).
  2. Intra-cyber (NEW): bottleneck autoencoder over cyber features, fitted
     on Normal windows only.  Residual = reconstruction error per step.
     Captures cyber feature correlation anomalies (e.g. port-scan produces
     unusual traffic patterns that break normal correlations even when no
     physical change occurs).

ICS-Flow weakness diagnosis: port-scan is a pure network attack; physical
sensors barely change.  Intra-physical residual is near zero, so the model
has no discriminative signal.  Intra-cyber residual fills this gap.

IGCPS/WDT: physical signals dominate; intra-cyber adds modest redundant
information.  lambda_cyber controls its weight.

Cross-domain direction is removed (CPCD3-w/o-cross was the best config);
this model is the next iteration of that baseline.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


CPCD4_VARIANTS: Dict[str, Dict[str, bool]] = {
    "CPCD4": {},
    # only intra-physical (= CPCD3-w/o-cross with new architecture)
    "CPCD4-w/o-cyber": {"no_cyber": True},
    # only intra-cyber (no physical consistency)
    "CPCD4-w/o-phys": {"no_phys": True},
    # neither: raw features only
    "CPCD4-w/o-consistency": {"no_cyber": True, "no_phys": True},
}


def split_schema(feature_names: Sequence[str], with_obs: bool = False):
    phys_base, cyber_like, target = [], [], []
    phys_all = [i for i, n in enumerate(feature_names) if n.startswith("phy_")]
    for i, n in enumerate(feature_names):
        if n.startswith("phy_dev_"):
            continue
        if n.startswith("phy_"):
            phys_base.append(i)
            if "_delta_observation_" not in n:
                target.append(i)
        else:
            cyber_like.append(i)
    if with_obs:
        return cyber_like, phys_base, target, phys_all
    return cyber_like, phys_base, target


class CPCD4(nn.Module):
    def __init__(self, input_dim: int, n_classes: int, feature_names: Sequence[str],
                 d_model: int = 64, dropout: float = 0.1,
                 lambda_phys: float = 0.5, lambda_cyber: float = 0.3,
                 flags: Dict[str, bool] | None = None, **_: object):
        super().__init__()
        self.flags = dict(flags or {})
        self.lambda_phys = float(lambda_phys)
        self.lambda_cyber = float(lambda_cyber)

        cyb, phys, tgt, obs = split_schema(feature_names, with_obs=True)
        self.register_buffer("cyb_idx",  torch.tensor(cyb,  dtype=torch.long))
        self.register_buffer("phys_idx", torch.tensor(phys, dtype=torch.long))
        self.register_buffer("tgt_idx",  torch.tensor(tgt,  dtype=torch.long))
        self.register_buffer("obs_idx",  torch.tensor(obs,  dtype=torch.long))
        dc, dp, dt = len(cyb), len(phys), len(tgt)
        self.n_target = dt
        self.n_cyber = dc

        # ---- shared encoder: cyber features -> latent sequence ----
        # Used by both the classifier and the cyber bottleneck predictor.
        self.cyber_token = nn.Sequential(nn.Linear(dc, d_model), nn.GELU(),
                                         nn.LayerNorm(d_model))
        self.cyber_encoder = nn.GRU(d_model, d_model, batch_first=True)

        # ---- intra-physical predictor (leave-one-out) ----
        if dt > 1 and not self.flags.get("no_phys"):
            self.phys_proj  = nn.Sequential(nn.Linear(dt, d_model), nn.GELU(),
                                             nn.LayerNorm(d_model))
            self.phys_mu    = nn.Linear(d_model, 1)
            self.phys_logvar = nn.Linear(d_model, 1)
            loo = torch.ones(dt, dt) - torch.eye(dt)
            self.register_buffer("phys_loo", loo)

        # ---- intra-cyber bottleneck predictor ----
        # Bottleneck AE: dc -> d_model//2 -> dc.
        # Trained on Normal windows only (Normal-only NLL).
        # Residual = (cyber - decoded) normalized by predicted std.
        if not self.flags.get("no_cyber"):
            neck = max(d_model // 2, 8)
            self.cyber_enc_bn = nn.Sequential(nn.Linear(dc, neck), nn.GELU(),
                                               nn.LayerNorm(neck))
            self.cyber_mu_bn  = nn.Linear(neck, dc)
            self.cyber_lv_bn  = nn.Linear(neck, dc)

        # ---- classifier input ----
        # Always: encoder hidden (d_model) + obs (len(obs))
        # + phys residual (dt, zeroed when no_phys)
        # + cyber residual collapsed to d_model scalar summary (1 per step)
        phys_dim  = dt if dt > 1 else 0
        # cyber residual: summarised as mean-squared per step -> 1 dim
        cyber_dim = 1
        res_in = len(obs) + phys_dim + cyber_dim
        self.res_embed = nn.Sequential(nn.Linear(res_in, d_model), nn.GELU(),
                                       nn.LayerNorm(d_model))
        self.res_pool  = nn.GRU(d_model, d_model, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(2 * d_model, 2 * d_model), nn.GELU(), nn.LayerNorm(2 * d_model),
            nn.Dropout(dropout), nn.Linear(2 * d_model, n_classes))

    # ------------------------------------------------------------------
    def predictor_parameters(self) -> List[nn.Parameter]:
        params: List[nn.Parameter] = []
        if hasattr(self, "phys_mu"):
            params += list(self.phys_mu.parameters()) + list(self.phys_logvar.parameters())
        if hasattr(self, "cyber_mu_bn"):
            params += list(self.cyber_mu_bn.parameters()) + list(self.cyber_lv_bn.parameters())
        return params

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        cyber  = x.index_select(-1, self.cyb_idx)    # (B, T, dc)  all steps
        target = x.index_select(-1, self.tgt_idx)    # (B, T, dt)
        obs    = x.index_select(-1, self.obs_idx)    # (B, T, |obs|)

        # ---- shared encoder ----
        h, _ = self.cyber_encoder(self.cyber_token(cyber))  # (B, T, d_model)

        # ---- intra-physical prediction ----
        no_phys = self.flags.get("no_phys") or self.n_target <= 1 or not hasattr(self, "phys_mu")
        if no_phys:
            phys_residual = torch.zeros(*target.shape[:2], max(self.n_target, 1),
                                        device=x.device)
        else:
            B, T, dt = target.shape
            masked = target.unsqueeze(2) * self.phys_loo      # (B, T, dt, dt)
            flat   = masked.reshape(B * T * dt, dt)
            hi     = self.phys_proj(flat)
            pm     = self.phys_mu(hi).reshape(B, T, dt)
            plv    = self.phys_logvar(hi).reshape(B, T, dt).clamp(-6.0, 6.0)
            phys_residual = ((target - pm) * torch.exp(-0.5 * plv)).clamp(-20.0, 20.0)

        # ---- intra-cyber bottleneck ----
        no_cyber = self.flags.get("no_cyber") or not hasattr(self, "cyber_mu_bn")
        if no_cyber:
            cyber_res_scalar = torch.zeros(*cyber.shape[:2], 1, device=x.device)
            cyber_mu_full  = torch.zeros_like(cyber)
            cyber_lv_full  = torch.zeros_like(cyber)
        else:
            B, T, dc = cyber.shape
            flat_c = cyber.reshape(B * T, dc)
            enc_c  = self.cyber_enc_bn(flat_c)
            cm     = self.cyber_mu_bn(enc_c).reshape(B, T, dc)
            clv    = self.cyber_lv_bn(enc_c).reshape(B, T, dc).clamp(-6.0, 6.0)
            cyber_mu_full = cm
            cyber_lv_full = clv
            cyber_residual = ((cyber - cm) * torch.exp(-0.5 * clv)).clamp(-20.0, 20.0)
            # collapse to 1-dim per step: mean squared normalized residual
            cyber_res_scalar = (cyber_residual ** 2).mean(dim=-1, keepdim=True)  # (B,T,1)

        # ---- classifier ----
        evidence = torch.cat([obs, phys_residual, cyber_res_scalar], dim=-1)
        r, _ = self.res_pool(self.res_embed(evidence))
        logits = self.classifier(torch.cat([h[:, -1], r[:, -1]], dim=-1))

        return {
            "logits": logits,
            "phys_target": target,
            "phys_residual": phys_residual,
            "cyber_mu": cyber_mu_full,
            "cyber_lv": cyber_lv_full,
            "cyber_target": cyber,
            "reconstruction": x[:, -1],
            "consistency_score": cyber_res_scalar[:, -1, 0],
        }

    # ------------------------------------------------------------------
    def _nll_phys(self, out, mask):
        if not hasattr(self, "phys_mu") or self.flags.get("no_phys"):
            return torch.tensor(0.0, device=out["logits"].device)
        # We need pm/plv stored; recompute from stored residual is not
        # straightforward, so we store them in forward and pass via out.
        # For simplicity recompute:
        return torch.tensor(0.0, device=out["logits"].device)  # placeholder

    def nll_phys_direct(self, x, y):
        """Compute intra-physical NLL directly (called from loss())."""
        target = x.index_select(-1, self.tgt_idx)
        if not hasattr(self, "phys_mu") or self.flags.get("no_phys"):
            return torch.tensor(0.0, device=x.device)
        B, T, dt = target.shape
        masked = target.unsqueeze(2) * self.phys_loo
        flat   = masked.reshape(B * T * dt, dt)
        hi     = self.phys_proj(flat)
        pm     = self.phys_mu(hi).reshape(B, T, dt)
        plv    = self.phys_logvar(hi).reshape(B, T, dt).clamp(-6.0, 6.0)
        per = 0.5 * (plv + (target - pm) ** 2 * torch.exp(-plv))
        per = per.mean(dim=(1, 2))
        mask = (y == 0)
        if not mask.any():
            return per.sum() * 0.0
        return per[mask].mean()

    def nll_cyber_direct(self, x, y):
        """Compute intra-cyber bottleneck NLL directly (called from loss())."""
        if not hasattr(self, "cyber_mu_bn") or self.flags.get("no_cyber"):
            return torch.tensor(0.0, device=x.device)
        cyber = x.index_select(-1, self.cyb_idx)
        B, T, dc = cyber.shape
        flat_c = cyber.reshape(B * T, dc)
        enc_c  = self.cyber_enc_bn(flat_c)
        cm  = self.cyber_mu_bn(enc_c).reshape(B, T, dc)
        clv = self.cyber_lv_bn(enc_c).reshape(B, T, dc).clamp(-6.0, 6.0)
        per = 0.5 * (clv + (cyber - cm) ** 2 * torch.exp(-clv))
        per = per.mean(dim=(1, 2))
        mask = (y == 0)
        if not mask.any():
            return per.sum() * 0.0
        return per[mask].mean()

    def loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        out = self(x)
        ce = F.cross_entropy(out["logits"], y, reduction="none")
        pt = torch.softmax(out["logits"], dim=-1).gather(1, y[:, None]).squeeze(1)
        loss = (((1.0 - pt) ** 2) * ce).mean()   # focal
        if not self.flags.get("no_phys"):
            loss = loss + self.lambda_phys * self.nll_phys_direct(x, y)
        if not self.flags.get("no_cyber"):
            loss = loss + self.lambda_cyber * self.nll_cyber_direct(x, y)
        return loss


def build_cpcd4(model_key: str, bundle, config, **hp):
    flags = CPCD4_VARIANTS[model_key.split("@")[0]]
    return CPCD4(input_dim=bundle.model_input_dim, n_classes=len(bundle.class_names),
                 feature_names=bundle.feature_names, flags=flags, **hp)
