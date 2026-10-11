"""What one training run does. Kaggle job configs are lists of recipes (one per GPU)."""

from dataclasses import dataclass, field

PE_L = "vit_pe_core_large_patch14_336.fb"


@dataclass(frozen=True)
class Recipe:
    name: str  # names the outputs: student_<name>.pt, scores_<name>.npz
    repeats: dict[str, int] = field(default_factory=dict)  # rows per source x n; 0 drops it
    arch: str = PE_L
    size: int = 336
    blocks: int = 8  # last blocks fine-tuned (plus attention pooling, final norm, head)
    steps: int = 3000
    batch: int = 32  # balanced: half AI, half real
    lr: float = 2e-5
    head_lr: float = 1e-3
    weight_decay: float = 0.05
