"""Round-3 CAFOAD variants for retrained (from-initialisation) ablation.

``CAFOADVariant`` subclasses the submitted ``CAFOAD`` without modifying it.
Each flag removes one path *before training*, so every reduced architecture is
optimised from scratch under the same folds, budget, and stopping rule.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

from common import revision

CAFOAD = revision.CAFOAD

# name -> flags removed before training
VARIANTS: Dict[str, Dict[str, bool]] = {
    "CAFOAD-full": {},
    "w/o-CAF": {"no_caf": True},
    "w/o-Transformer": {"no_transformer": True},
    "w/o-residual": {"no_residual": True},
    "w/o-domain-summary": {"no_domain_summary": True},
    "w/o-prototype": {"no_prototype": True},
    "w/o-SupCon": {"no_supcon": True},
    # Only the CAF -> Transformer -> fusion head path (no statistical skips).
    "CAF-path-only": {"no_residual": True, "no_domain_summary": True},
}


class CAFOADVariant(CAFOAD):
    def __init__(self, *args, flags: Dict[str, bool] | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.flags = dict(flags or {})

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        f = self.flags
        cyber = self.cyber_encoder(x.index_select(-1, self.cyber_indices))
        physical = self.physical_encoder(x.index_select(-1, self.physical_indices))
        domain_summary = torch.cat(
            [
                cyber[:, -1], cyber.mean(dim=1), cyber[:, -1] - cyber[:, 0],
                physical[:, -1], physical.mean(dim=1), physical[:, -1] - physical[:, 0],
            ],
            dim=-1,
        )
        if f.get("no_caf"):
            # Same width, no cross-domain exchange: average of the two encodings.
            fused = self.input_projection(self.fusion_norm(0.5 * (cyber + physical)))
        else:
            p_to_c, _ = self.cross_p_to_c(physical, cyber, cyber, need_weights=False)
            c_to_p, _ = self.cross_c_to_p(cyber, physical, physical, need_weights=False)
            gate = self.gate(torch.cat([p_to_c, c_to_p], dim=-1))
            fused = self.input_projection(
                self.fusion_norm(gate * p_to_c + (1.0 - gate) * c_to_p)
            )
        if not f.get("no_transformer"):
            length = fused.shape[1]
            mask = torch.triu(
                torch.ones(length, length, device=x.device, dtype=torch.bool), diagonal=1
            )
            fused = self.temporal_encoder(fused + self.position[:, :length], mask=mask)
        weights = torch.softmax(self.pool_score(fused).squeeze(-1), dim=1)
        pooled = torch.sum(fused * weights.unsqueeze(-1), dim=1)
        representation = torch.cat([fused[:, -1], pooled], dim=-1)
        base_logits = self.fusion_classifier(representation)
        if not f.get("no_residual"):
            residual = self.residual_context(x)
            base_logits = (
                base_logits + self.input_classifier(residual) + self.input_nonlinear(residual)
            )
        if not f.get("no_domain_summary"):
            base_logits = base_logits + self.domain_summary_classifier(domain_summary)
        projection = F.normalize(self.projection_head(fused[:, -1]), dim=-1)
        prototype_logits = (
            projection @ F.normalize(self.class_prototypes, dim=-1).T
        ) / self.prototype_temperature.clamp_min(1.0e-3)
        logits = base_logits + self.prototype_logit_weight * prototype_logits
        return {
            "logits": logits,
            "base_logits": base_logits,
            "prototype_logits": prototype_logits,
            "fusion_logits": logits,
            "projection": projection,
            "prediction": F.normalize(self.prediction_head(projection), dim=-1),
            "reconstruction": self.decoder(fused[:, -1]),
        }
