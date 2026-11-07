"""CPCD3: cross-domain + intra-physical predictive consistency.

Single mechanism (predictive consistency, Normal-only NLL), two directions:
  1. Cross-domain (CPCD2): predict x^p_t from cyber_t + physical_{t-1}.
     Residual = departure from legitimate cyber->physical coupling.
  2. Intra-physical (NEW): predict each physical sensor from the OTHER physical
     sensors at the same step.  Residual = inter-sensor inconsistency.

The intra-physical direction directly addresses the zero-order-hold problem:
when physical values are held constant across grid endpoints, the cross-domain
residual is zero, but an attack that breaks sensor correlations (e.g. spoofing
one sensor while others reflect the real process) produces a non-zero
intra-physical residual even though no temporal change is visible.

Both predictors are fitted with Normal-only NLL, so both residuals measure
departure from the legitimate coupling learned from training-Normal operation.
The classifier sees the encoder state, the observed physical state (incl.
deviation), the cross-domain residual, and the intra-physical residual.

This is the SAME mechanism family as CPCD2 -- predictive consistency under a
Normal-coupling assumption -- applied in a second direction, not a stacked
module of a different type.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

# Variants removed / changed *before training* (retrained ablations).
CPCD3_VARIANTS: Dict[str, Dict[str, bool]] = {
    "CPCD3": {},
    # no intra-physical predictor: isolates its contribution
    "CPCD3-w/o-intra": {"no_intra": True},
    # no cross-domain predictor: only intra-physical consistency
    "CPCD3-w/o-cross": {"no_cross": True},
    # no consistency at all: classifier gets raw physical levels only
    "CPCD3-w/o-consistency": {"no_cross": True, "no_intra": True},
}


def split_schema(feature_names: Sequence[str], with_obs: bool = False):
    """Return index lists (cyber_like, phys_base, phys_target[, phys_all]).

    phys_target = physical level coordinates (no delta/deviation suffix).
    """
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


class CPCD3(nn.Module):
    def __init__(self, input_dim: int, n_classes: int, feature_names: Sequence[str],
                 d_model: int = 64, dropout: float = 0.1, lambda_nll: float = 0.5,
                 lambda_intra: float = 0.3, detach_residual: bool = False,
                 flags: Dict[str, bool] | None = None, **_: object):
        super().__init__()
        self.flags = dict(flags or {})
        self.lambda_nll = float(lambda_nll)
        self.lambda_intra = float(lambda_intra)
        self.detach_residual = bool(detach_residual)
        cyb, phys, tgt, obs = split_schema(feature_names, with_obs=True)
        if not tgt:
            raise ValueError("CPCD3 needs at least one physical level coordinate")
        self.register_buffer("cyb_idx", torch.tensor(cyb, dtype=torch.long))
        self.register_buffer("phys_idx", torch.tensor(phys, dtype=torch.long))
        self.register_buffer("tgt_idx", torch.tensor(tgt, dtype=torch.long))
        self.register_buffer("obs_idx", torch.tensor(obs, dtype=torch.long))
        dc, dp, dt = len(cyb), len(phys), len(tgt)
        self.n_target = dt

        # ---- cross-domain encoder (cyber_t + physical_{t-1} -> predict x^p_t) ----
        token_in = dc + dp
        self.token = nn.Sequential(nn.Linear(token_in, d_model), nn.GELU(),
                                   nn.LayerNorm(d_model))
        self.encoder = nn.GRU(d_model, d_model, batch_first=True)
        self.mu_head = nn.Linear(d_model, dt)
        self.logvar_head = nn.Linear(d_model, dt)

        # ---- intra-physical predictor (predict each sensor from the others) ----
        # Input: all physical level coords except the one being predicted.
        # For sensor i: zero out column i, then project (B*T, dt) -> (B*T, d_model)
        # -> predict scalar mu_i and logvar_i.  Output: (B, T, dt).
        if dt > 1 and not self.flags.get("no_intra"):
            self.intra_proj = nn.Sequential(nn.Linear(dt, d_model), nn.GELU(),
                                             nn.LayerNorm(d_model))
            self.intra_mu = nn.Linear(d_model, 1)
            self.intra_logvar = nn.Linear(d_model, 1)
            # leave_one_out_mask[i, :] = 1 everywhere except position i
            loo = torch.ones(dt, dt) - torch.eye(dt)
            self.register_buffer("intra_loo", loo)  # (dt, dt)

        # ---- classifier evidence ----
        # evidence = observed physical (obs) + cross-domain residual + intra residual
        # intra dimension is always allocated (dt when dt>1, else 0) so that the
        # res_embed width is constant across ablation variants; ablations zero
        # the corresponding residual.
        intra_dim = dt if dt > 1 else 0
        res_in = len(obs) + dt + intra_dim
        self.res_embed = nn.Sequential(nn.Linear(res_in, d_model), nn.GELU(),
                                      nn.LayerNorm(d_model))
        self.res_pool = nn.GRU(d_model, d_model, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(2 * d_model, 2 * d_model), nn.GELU(), nn.LayerNorm(2 * d_model),
            nn.Dropout(dropout), nn.Linear(2 * d_model, n_classes))

    # ------------------------------------------------------------------
    def predictor_parameters(self) -> List[nn.Parameter]:
        """Parameters adapted online (normal-coupling heads only)."""
        params = list(self.mu_head.parameters()) + list(self.logvar_head.parameters())
        if hasattr(self, "intra_mu"):
            params += list(self.intra_mu.parameters())
            params += list(self.intra_logvar.parameters())
        return params

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        cyber = x[:, 1:].index_select(-1, self.cyb_idx)          # steps 2..q
        prev_phys = x[:, :-1].index_select(-1, self.phys_idx)    # steps 1..q-1
        target = x[:, 1:].index_select(-1, self.tgt_idx)         # physical level at 2..q
        obs = x[:, 1:].index_select(-1, self.obs_idx)            # all physical incl delta

        # ---- cross-domain prediction ----
        no_cross = self.flags.get("no_cross")
        if no_cross:
            mu = torch.zeros_like(target)
            logvar = torch.zeros_like(target)
            cd_residual = torch.zeros_like(target)
        else:
            tokens = torch.cat([cyber, prev_phys], dim=-1)
            h, _ = self.encoder(self.token(tokens))
            mu = self.mu_head(h)
            logvar = self.logvar_head(h).clamp(-6.0, 6.0)
            m, lv = (mu.detach(), logvar.detach()) if self.detach_residual else (mu, logvar)
            cd_residual = ((target - m) * torch.exp(-0.5 * lv)).clamp(-20.0, 20.0)

        # ---- intra-physical prediction ----
        no_intra = self.flags.get("no_intra") or self.n_target <= 1 or not hasattr(self, "intra_mu")
        if no_intra:
            intra_residual = torch.zeros_like(target)
            intra_mu = torch.zeros_like(target)
            intra_logvar = torch.zeros_like(target)
        else:
            # Leave-one-out: for each sensor i, predict it from all others.
            # target: (B, T, dt).  Build (B, T, dt, dt) where [b,t,i,:] = target[b,t,:]*loo[i,:]
            B, T, dt = target.shape
            # Expand for broadcasting: (B, T, 1, dt) * (dt, dt) -> (B, T, dt, dt)
            masked = target.unsqueeze(2) * self.intra_loo  # (B, T, dt, dt)
            flat = masked.reshape(B * T * dt, dt)
            hi = self.intra_proj(flat)                     # (B*T*dt, d_model)
            intra_mu = self.intra_mu(hi).reshape(B, T, dt)
            intra_logvar = self.intra_logvar(hi).reshape(B, T, dt).clamp(-6.0, 6.0)
            im, ilv = ((intra_mu.detach(), intra_logvar.detach())
                       if self.detach_residual else (intra_mu, intra_logvar))
            intra_residual = ((target - im) * torch.exp(-0.5 * ilv)).clamp(-20.0, 20.0)

        # ---- classifier evidence ----
        # Encoder always runs so h is available for the classifier head.
        if no_cross:
            tokens = torch.cat([cyber, prev_phys], dim=-1)
            h, _ = self.encoder(self.token(tokens))
        evidence = torch.cat([obs, cd_residual, intra_residual], dim=-1)
        r, _ = self.res_pool(self.res_embed(evidence))
        logits = self.classifier(torch.cat([h[:, -1], r[:, -1]], dim=-1))

        return {
            "logits": logits,
            "mu": mu, "logvar": logvar, "target": target,
            "residual": cd_residual,
            "intra_mu": intra_mu, "intra_logvar": intra_logvar,
            "intra_residual": intra_residual,
            "reconstruction": x[:, -1],  # predict_loader computes recon error
            "consistency_score": 0.5 * (cd_residual[:, -1] ** 2).mean(dim=-1),
        }

    def nll_cross(self, out: Dict[str, torch.Tensor], mask: torch.Tensor | None) -> torch.Tensor:
        per = 0.5 * (out["logvar"] + (out["target"] - out["mu"]) ** 2 * torch.exp(-out["logvar"]))
        per = per.mean(dim=(1, 2))
        if mask is not None:
            if not mask.any():
                return per.sum() * 0.0
            per = per[mask]
        return per.mean()

    def nll_intra(self, out: Dict[str, torch.Tensor], mask: torch.Tensor | None) -> torch.Tensor:
        if not hasattr(self, "intra_mu") or self.flags.get("no_intra"):
            return torch.tensor(0.0, device=out["logits"].device)
        per = 0.5 * (out["intra_logvar"] +
                     (out["target"] - out["intra_mu"]) ** 2 * torch.exp(-out["intra_logvar"]))
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
        loss = (((1.0 - pt) ** 2) * ce).mean()          # focal loss
        normal_mask = (y == 0)
        if not self.flags.get("no_cross"):
            loss = loss + self.lambda_nll * self.nll_cross(out, normal_mask)
        if not self.flags.get("no_intra"):
            loss = loss + self.lambda_intra * self.nll_intra(out, normal_mask)
        return loss


def build_cpcd3(model_key: str, bundle, config, **hp):
    flags = CPCD3_VARIANTS[model_key.split("@")[0]]
    return CPCD3(input_dim=bundle.model_input_dim, n_classes=len(bundle.class_names),
                feature_names=bundle.feature_names, flags=flags, **hp)
