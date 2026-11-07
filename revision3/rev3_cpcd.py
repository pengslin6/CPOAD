"""Cross-domain Predictive Consistency Detector (CPCD), round 3.

Single mechanism: a normal-coupling predictor estimates the current physical
state from current cyber evidence and the *previous* physical state.  The
classifier sees the encoder state and the standardized cross-domain
prediction residual.  The predictor is fitted (NLL) on Normal windows only,
so the residual measures departure from the legitimate cyber->physical
coupling learned from training-Normal operation.

Leakage rule: the physical value at step t is never an input of its own
prediction (only physical rows up to t-1 and the cyber row at t).  Physical
delta/deviation columns at step t are excluded for the same reason.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

# Variants removed / changed *before training* (retrained ablations).
CPCD_VARIANTS: Dict[str, Dict[str, bool]] = {
    "CPCD": {},
    # no predictor, no NLL: classifier gets standardized raw physical levels
    "CPCD-w/o-consistency": {"no_consistency": True},
    # predictor uses physical history only (no cross-domain information)
    "CPCD-phys-only-predictor": {"phys_only_predictor": True},
    # predictor NLL on all classes instead of Normal only
    "CPCD-all-class-predictor": {"all_class_predictor": True},
    # v2 (dev round 2): classifier always sees the observed physical state at t;
    # the residual is concatenated, so the ablation isolates the residual only.
    "CPCD2": {"v2": True},
    "CPCD2-cyber-predictor": {"v2": True, "cyber_only_predictor": True},
    "CPCD2-w/o-consistency": {"v2": True, "no_consistency": True},
}


def split_schema(feature_names: Sequence[str], with_obs: bool = False):
    """Return index lists (cyber_like, phys_base, phys_target[, phys_all])."""
    phys_base, cyber_like, target = [], [], []
    phys_all = [i for i, n in enumerate(feature_names) if n.startswith("phy_")]
    for i, n in enumerate(feature_names):
        if n.startswith("phy_dev_"):
            continue  # predictor never sees the deviation of the current value
        if n.startswith("phy_"):
            phys_base.append(i)
            if "_delta_observation_" not in n:
                target.append(i)
        else:
            cyber_like.append(i)  # cyber_, cyber_dev_, age/changed flags
    if with_obs:
        return cyber_like, phys_base, target, phys_all
    return cyber_like, phys_base, target


class CPCD(nn.Module):
    def __init__(self, input_dim: int, n_classes: int, feature_names: Sequence[str],
                 d_model: int = 64, dropout: float = 0.1, lambda_nll: float = 0.5,
                 detach_residual: bool = False, flags: Dict[str, bool] | None = None,
                 **_: object):
        super().__init__()
        self.flags = dict(flags or {})
        self.lambda_nll = float(lambda_nll)
        self.detach_residual = bool(detach_residual)
        cyb, phys, tgt, obs = split_schema(feature_names, with_obs=True)
        if not tgt:
            raise ValueError("CPCD needs at least one physical level coordinate")
        self.register_buffer("cyb_idx", torch.tensor(cyb, dtype=torch.long))
        self.register_buffer("phys_idx", torch.tensor(phys, dtype=torch.long))
        self.register_buffer("tgt_idx", torch.tensor(tgt, dtype=torch.long))
        self.register_buffer("obs_idx", torch.tensor(obs, dtype=torch.long))
        dc, dp, dt = len(cyb), len(phys), len(tgt)
        self.n_target = dt
        self.v2 = bool(self.flags.get("v2"))
        self.cyber_only = bool(self.flags.get("cyber_only_predictor"))
        # shared causal encoder over tokens [cyber_t ; physical_{t-1}]
        token_in = dc if self.cyber_only else dc + dp
        self.token = nn.Sequential(nn.Linear(token_in, d_model), nn.GELU(), nn.LayerNorm(d_model))
        self.encoder = nn.GRU(d_model, d_model, batch_first=True)
        if self.flags.get("phys_only_predictor"):
            self.phys_token = nn.Sequential(nn.Linear(dp, d_model), nn.GELU(), nn.LayerNorm(d_model))
            self.phys_encoder = nn.GRU(d_model, d_model, batch_first=True)
        # normal-coupling predictor heads (mean and log-variance of x^p_t)
        self.mu_head = nn.Linear(d_model, dt)
        self.logvar_head = nn.Linear(d_model, dt)
        res_in = len(obs) + dt if self.v2 else dt
        self.res_embed = nn.Sequential(nn.Linear(res_in, d_model), nn.GELU(), nn.LayerNorm(d_model))
        self.res_pool = nn.GRU(d_model, d_model, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(2 * d_model, 2 * d_model), nn.GELU(), nn.LayerNorm(2 * d_model),
            nn.Dropout(dropout), nn.Linear(2 * d_model, n_classes))

    # ------------------------------------------------------------------
    def predictor_parameters(self) -> List[nn.Parameter]:
        """Parameters adapted online (normal-coupling heads only)."""
        return list(self.mu_head.parameters()) + list(self.logvar_head.parameters())

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        cyber = x[:, 1:].index_select(-1, self.cyb_idx)          # steps 2..q
        prev_phys = x[:, :-1].index_select(-1, self.phys_idx)    # steps 1..q-1
        target = x[:, 1:].index_select(-1, self.tgt_idx)         # physical level at 2..q
        tokens = cyber if self.cyber_only else torch.cat([cyber, prev_phys], dim=-1)
        h, _ = self.encoder(self.token(tokens))
        if self.flags.get("no_consistency") and self.v2:
            mu = torch.zeros_like(target)
            logvar = torch.zeros_like(target)
            residual = torch.zeros_like(target)
        elif self.flags.get("no_consistency"):
            mu = torch.zeros_like(target)
            logvar = torch.zeros_like(target)
            residual = target
        else:
            hp = h
            if self.flags.get("phys_only_predictor"):
                hp, _ = self.phys_encoder(self.phys_token(prev_phys))
            mu = self.mu_head(hp)
            logvar = self.logvar_head(hp).clamp(-6.0, 6.0)
            m, lv = (mu.detach(), logvar.detach()) if self.detach_residual else (mu, logvar)
            residual = ((target - m) * torch.exp(-0.5 * lv)).clamp(-20.0, 20.0)
        evidence = residual
        if self.v2:
            evidence = torch.cat([x[:, 1:].index_select(-1, self.obs_idx), residual], dim=-1)
        r, _ = self.res_pool(self.res_embed(evidence))
        logits = self.classifier(torch.cat([h[:, -1], r[:, -1]], dim=-1))
        return {
            "logits": logits, "mu": mu, "logvar": logvar, "target": target,
            "residual": residual,
            # predict_loader computes ((reconstruction - x[:, -1])**2): we expose
            # the consistency score through a tensor of the same shape.
            "reconstruction": x[:, -1],
            "consistency_score": 0.5 * (residual[:, -1] ** 2).mean(dim=-1),
        }

    def nll(self, out: Dict[str, torch.Tensor], mask: torch.Tensor | None = None) -> torch.Tensor:
        per = 0.5 * (out["logvar"] + (out["target"] - out["mu"]) ** 2 * torch.exp(-out["logvar"]))
        per = per.mean(dim=(1, 2))
        if mask is not None:
            if not mask.any():
                return per.sum() * 0.0
            per = per[mask]
        return per.mean()

    def loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        out = self(x)
        ce = F.cross_entropy(out["logits"], y, reduction="none")
        pt = torch.softmax(out["logits"], dim=-1).gather(1, y[:, None]).squeeze(1)
        loss = (((1.0 - pt) ** 2) * ce).mean()          # same focal loss as baselines
        if not self.flags.get("no_consistency"):
            mask = None if self.flags.get("all_class_predictor") else (y == 0)
            loss = loss + self.lambda_nll * self.nll(out, mask)
        return loss


def build_cpcd(model_key: str, bundle, config, **hp):
    flags = CPCD_VARIANTS[model_key.split("@")[0]]
    return CPCD(input_dim=bundle.model_input_dim, n_classes=len(bundle.class_names),
                feature_names=bundle.feature_names, flags=flags, **hp)
