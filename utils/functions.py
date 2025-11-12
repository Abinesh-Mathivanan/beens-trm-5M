import importlib

def load_model_class(identifier: str, prefix: str = "models."):
    module_path, class_name = identifier.split('@')
    return getattr(importlib.import_module(prefix + module_path), class_name)