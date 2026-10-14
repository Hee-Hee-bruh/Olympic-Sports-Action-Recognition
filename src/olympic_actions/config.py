from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class Config:
    samples: int = 10
    image_size: int = 224
    seed: int = 42
    validation_fraction: float = 0.2
    svm_c: tuple = (0.01, 0.1, 1.0, 10.0)
    svm_max_iter: int = 20000
    device: str = "cpu"
    resnet_batch_size: int = 10
    resnet_weights: str = "IMAGENET1K_V1"
    hog_orientations: int = 9
    hog_cell_size: int = 16
    hog_block_cells: int = 2
    flow_grid: int = 2
    flow_bins: int = 8
    flow_pyr_scale: float = 0.5
    flow_levels: int = 3
    flow_winsize: int = 15
    flow_iterations: int = 3
    flow_poly_n: int = 5
    flow_poly_sigma: float = 1.2

    def __post_init__(self):
        for name in ("samples", "image_size", "resnet_batch_size", "svm_max_iter",
                     "hog_cell_size", "hog_block_cells", "hog_orientations",
                     "flow_grid", "flow_bins", "flow_levels", "flow_iterations",
                     "flow_winsize"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.image_size // self.hog_cell_size < self.hog_block_cells:
            raise ValueError("Image is too small for HOG blocks")
        if self.image_size < self.flow_grid:
            raise ValueError("Image is too small for the flow grid")
        if not 0 < self.validation_fraction < 1:
            raise ValueError("validation_fraction must be between zero and one")
        if not self.svm_c or any(c <= 0 for c in self.svm_c):
            raise ValueError("svm_c must contain positive values")
        if self.resnet_weights != "IMAGENET1K_V1":
            raise ValueError("This experiment requires frozen IMAGENET1K_V1 weights")
        if self.flow_poly_n not in (5, 7) or not 0 < self.flow_pyr_scale < 1:
            raise ValueError("Invalid Farneback settings")

    @classmethod
    def load(cls, path=None):
        if path is None:
            return cls()
        values = json.loads(Path(path).read_text())
        if "svm_c" in values:
            values["svm_c"] = tuple(values["svm_c"])
        return cls(**values)

    def to_dict(self):
        return asdict(self)
