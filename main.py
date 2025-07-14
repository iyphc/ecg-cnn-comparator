import hydra
import os
from omegaconf import DictConfig, OmegaConf
import torch
from src.data.loader import get_dataloaders
from src.training.evaluator import compare_models, basic_scores, evaluate_model
from src.training.trainer import train_model

@hydra.main(version_base="1.2", config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print("Используемая конфигурация:")
    print(OmegaConf.to_yaml(cfg))

    train_loader, test_loader, valid_loader, class_names, features_num = get_dataloaders(
        batch_size=cfg.data.batch_size,
        valid_part=cfg.data.val_part,
        num_workers=cfg.data.num_workers
    )
    out_classes = len(class_names)

    if cfg.mode == "compare":
        base_cfg = OmegaConf.load("configs/model/base_model.yaml")
        handcrafted_cfg = OmegaConf.load("configs/model/handcrafted.yaml")

        base_model = hydra.utils.instantiate(base_cfg.model)
        handcrafted_model = hydra.utils.instantiate(handcrafted_cfg.model, base_model=base_model)

        save_dir = "models/checkpoints"
        os.makedirs(save_dir, exist_ok=True)
        
        if cfg.train or not os.path.exists(f"{save_dir}/handcrafted_CNN_ECG_detection.pth"):
            print("Обучение handcrafted модели...")
            handcrafted_model = train_model(
                handcrafted_model, train_loader, test_loader, valid_loader, class_names,
                is_handcrafted=True, epochs=cfg.training.epochs, batch_size=cfg.data.batch_size, learning_rate=cfg.training.lr
            )
            torch.save(handcrafted_model.state_dict(), f"{save_dir}/handcrafted_CNN_ECG_detection.pth")
        else:
            handcrafted_model.load_state_dict(torch.load(f"{save_dir}/handcrafted_CNN_ECG_detection.pth"))
            print("HANDCRAFTED MODEL IS LOADED")

        if cfg.train or not os.path.exists(f"{save_dir}/CNN_ECG_detection.pth"):
            print("Обучение базовой модели...")
            base_model = train_model(
                base_model, train_loader, test_loader, valid_loader, class_names,
                is_handcrafted=False, epochs=cfg.training.epochs, batch_size=cfg.data.batch_size, learning_rate=cfg.training.lr
            )
            torch.save(base_model.state_dict(), f"{save_dir}/CNN_ECG_detection.pth")
        else:
            base_model.load_state_dict(torch.load(f"{save_dir}/CNN_ECG_detection.pth"))
            print("BASE MODEL IS LOADED")

        for model, name in [(base_model, "BASE"), (handcrafted_model, "HANDCRAFTED")]:
            all_preds, all_true = evaluate_model(model, test_loader, is_handcrafted=(name == "HANDCRAFTED"))
            scores = basic_scores(all_true, all_preds)
            print(f"\n\n-----------------{name} SCORES-----------------\n")
            for score_name, score_value in scores.items():
                print(f"{score_name}: {score_value}")

        intervals = compare_models(base_model, handcrafted_model, test_loader)
        print("\n\n--------------------COMPARISON INTERVALS--------------------\n")
        for name, interval in intervals.items():
            print(f"{name}: ({interval[0]:.4f}, {interval[1]:.4f})")

    elif cfg.mode == "train":
        model = hydra.utils.instantiate(cfg.model)
        save_path = os.path.join(cfg.training.save_path, cfg.training.save_name)
        model = train_model(
            model, train_loader, test_loader, valid_loader, class_names,
            is_handcrafted=cfg.handcrafted, epochs=cfg.training.epochs,
            batch_size=cfg.data.batch_size, learning_rate=cfg.training.lr
        )
        os.makedirs(os.path.dirname(cfg.training.save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"Модель сохранена по пути: {cfg.training.save_path}")

    elif cfg.mode == "statistics":
        model = hydra.utils.instantiate(cfg.model)
        save_path = os.path.join(cfg.training.save_path, cfg.training.save_name)
        if cfg.handcrafted:
            save_path = "handcrafted_" + save_path

        if os.path.exists(save_path):
            model.load_state_dict(torch.load(save_path))
            print(f"MODEL IS LOADED FROM {save_path}")
        else:
            print(f"Model not found at {save_path}, training...")
            train_model(model, train_loader, test_loader, valid_loader, class_names,
                        is_handcrafted=cfg.handcrafted, epochs=cfg.training.epochs,
                        batch_size=cfg.data.batch_size, learning_rate=cfg.training.lr)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"Model saved to {save_path}")

        all_preds, all_true = evaluate_model(model, test_loader, is_handcrafted=cfg.handcrafted)
        scores = basic_scores(all_true, all_preds)
        print("\n\n--------------------STATISTICS--------------------\n")
        for name, score in scores.items():
            print(f"{name}: {score}")

    else:
        print("Неверный режим. Доступные режимы: train, compare, statistics")

if __name__ == "__main__":
    main()