import hydra
import os
from omegaconf import DictConfig, OmegaConf
import torch
from src.ecg_analyzer.utils.handlers import handler_train
from src.ecg_analyzer.utils.handlers import handler_compare
from src.ecg_analyzer.utils.handlers import handler_evaluate

@hydra.main(version_base="1.2", config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print("Используемая конфигурация:")
    print(OmegaConf.to_yaml(cfg))

    if cfg.mode == "compare":
        handler_compare(cfg)

    elif cfg.mode == "train": 
       handler_train(cfg)

    elif cfg.mode == "evaluate":
        handler_evaluate(cfg)
    else:
        print("Неверный режим. Доступные режимы: train, compare, statistics")

if __name__ == "__main__":
    main()