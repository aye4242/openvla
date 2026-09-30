try:
    from .models import available_model_names, available_models, get_model_description, load
except (ImportError, TypeError) as exc:
    # HuggingFace inference imports ``prismatic.extern.hf`` directly and does
    # not need the RLDS/dlimp training stack. Keep that lightweight path
    # usable when optional dataset dependencies are absent.
    if not any(
        token in str(exc)
        for token in ("dlimp", "tensorflow_datasets", "runtime_version", "built-in module")
    ):
        raise
    available_model_names = available_models = get_model_description = load = None
