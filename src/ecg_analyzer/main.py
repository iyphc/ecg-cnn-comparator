import hydra
from omegaconf import DictConfig, OmegaConf
from ecg_analyzer.utils.handlers import handler_train, handler_compare, handler_evaluate


@hydra.main(version_base="1.2", config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    print("=" * 50)

    if cfg.mode == "compare":
        handler_compare(cfg)
    elif cfg.mode == "train":
        handler_train(cfg)
    elif cfg.mode == "evaluate":
        handler_evaluate(cfg)
    else:
        print("Неверный режим. Доступные режимы: train, compare, evaluate")


if __name__ == "__main__":
    main()
