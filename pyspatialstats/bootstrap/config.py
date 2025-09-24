from dataclasses import dataclass, field

import numpy as np


@dataclass()
class BootstrapConfig:
    n_bootstraps: int = 1000
    seed: int = field(default_factory=lambda: np.random.randint(0, np.iinfo(np.int32).max))
