from __future__ import annotations

import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg: DictConfig) -> None:
    resolved = OmegaConf.to_container(cfg, resolve=True)
    demo_cfg = resolved["demo"]
    data_cfg = resolved["data"]
    print(f"Starting {demo_cfg['name']} on {demo_cfg['host']}:{demo_cfg['port']}")
    print(f"Classes: {', '.join(data_cfg['classes'])}")


if __name__ == "__main__":
    main()
