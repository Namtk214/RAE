"""Transport package — create_transport factory."""

from .transport import Transport, ModelType, WeightType, PathType, Sampler


def create_transport(
    path_type='Linear',
    prediction="velocity",
    loss_weight=None,
    train_eps=None,
    sample_eps=None,
    time_dist_type="uniform",
    time_dist_shift=1.0,
):
    """Create Transport object for flow matching.

    Args:
        path_type: 'Linear', 'GVP', or 'VP'
        prediction: 'velocity', 'noise', or 'score'
        loss_weight: 'velocity', 'likelihood', or None
        time_dist_type: 'uniform' or 'logit-normal_mu_sigma'
        time_dist_shift: >= 1.0
    """
    model_map = {"noise": ModelType.NOISE, "score": ModelType.SCORE, "velocity": ModelType.VELOCITY}
    model_type = model_map.get(prediction, ModelType.VELOCITY)

    loss_map = {"velocity": WeightType.VELOCITY, "likelihood": WeightType.LIKELIHOOD}
    loss_type = loss_map.get(loss_weight, WeightType.NONE)

    path_map = {"Linear": PathType.LINEAR, "GVP": PathType.GVP, "VP": PathType.VP}
    path_type = path_map[path_type]

    if path_type == PathType.VP:
        train_eps = train_eps or 1e-5
        sample_eps = sample_eps or 1e-3
    elif path_type in (PathType.GVP, PathType.LINEAR) and model_type != ModelType.VELOCITY:
        train_eps = train_eps or 1e-3
        sample_eps = sample_eps or 1e-3
    else:
        train_eps = train_eps or 0
        sample_eps = sample_eps or 0

    return Transport(
        model_type=model_type,
        path_type=path_type,
        loss_type=loss_type,
        time_dist_type=time_dist_type,
        time_dist_shift=time_dist_shift,
        train_eps=train_eps,
        sample_eps=sample_eps,
    )
