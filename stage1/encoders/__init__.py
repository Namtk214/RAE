from .dinov2 import Dinov2withNorm

ENCODER_REGISTRY = {
    'Dinov2withNorm': Dinov2withNorm,
}

def get_encoder(name: str):
    if name not in ENCODER_REGISTRY:
        raise ValueError(f"Unknown encoder: {name}. Available: {list(ENCODER_REGISTRY.keys())}")
    return ENCODER_REGISTRY[name]
