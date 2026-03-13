from omegaconf import DictConfig, OmegaConf

import hydra
from meta_learning.utils.logging import get_logger

logger = get_logger(__name__)


@hydra.main(version_base=None, config_path="config", config_name="config")
def run(cfg: DictConfig) -> None:
    logger.info("Config:\n%s", OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    run()
