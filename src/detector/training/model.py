"""The student: a timm PE-Core backbone and a linear head on its pooled embedding.

Checkpoints hold only what training changed ({"model": trained backbone params, fp16; "head":
linear head}); the rest comes from the public pretrained weights.
"""

from pathlib import Path

import timm
import torch

from detector.training.recipe import PE_L, Recipe


class Student(torch.nn.Module):
    def __init__(self, backbone: torch.nn.Module, head: torch.nn.Module) -> None:
        super().__init__()
        self.backbone, self.head = backbone, head

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """(n, 3, size, size) in [-1, 1] -> (n,) logits."""
        return self.head(self.backbone(images)).squeeze(1)

    @classmethod
    def for_training(cls, recipe: Recipe) -> "Student":
        """Frozen part in fp16 (fits a 16 GB T4); last blocks, pooling, norm and head in fp32."""
        backbone = timm.create_model(
            recipe.arch, pretrained=True, num_classes=0, img_size=recipe.size
        )
        backbone.half()
        backbone.set_grad_checkpointing(True)
        backbone.requires_grad_(False)
        for module in [backbone.blocks[-recipe.blocks :], backbone.attn_pool, backbone.norm]:
            module.float().requires_grad_(True)
        return cls(backbone, torch.nn.Linear(backbone.num_features, 1))

    @classmethod
    def from_checkpoint(cls, path: str | Path, arch: str = PE_L) -> "Student":
        """fp32 student with the trained weights, for export and laptop scoring."""
        backbone = timm.create_model(arch, pretrained=True, num_classes=0)
        saved = torch.load(path, map_location="cpu")
        weights = {k: v.float() for k, v in saved["model"].items()}
        unexpected = backbone.load_state_dict(weights, strict=False).unexpected_keys
        assert not unexpected, unexpected
        head = torch.nn.Linear(backbone.num_features, 1)
        head.load_state_dict({k: v.float() for k, v in saved["head"].items()})
        return cls(backbone, head).eval()

    def save(self, path: str | Path) -> None:
        trained = {n for n, p in self.backbone.named_parameters() if p.requires_grad}
        state = {k: v.half().cpu() for k, v in self.backbone.state_dict().items() if k in trained}
        torch.save({"model": state, "head": self.head.state_dict()}, path)
