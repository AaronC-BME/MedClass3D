from datetime import datetime

from omegaconf import DictConfig, OmegaConf


def make_omegaconf_resolvers():
    OmegaConf.register_new_resolver(
        "path_formatter",
        lambda s: s.replace("[", "")
        .replace("]", "")
        .replace("}", "")
        .replace("{", "")
        .replace(")", "")
        .replace("(", "")
        .replace(",", "_")
        .replace("=", "_")
        .replace("/", ".")
        .replace("+", "")
        .replace("@", "."),
    )
    OmegaConf.register_new_resolver("model_name_extractor", lambda s: s.split(".")[-1])
    OmegaConf.register_new_resolver(
        "make_group_name",
        lambda: datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        use_cache=True,
    )
