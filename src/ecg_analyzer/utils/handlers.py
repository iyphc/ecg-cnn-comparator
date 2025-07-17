import os 
import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from ..training.trainer import train_model
from ..training.evaluator import evaluate_model, basic_scores, compare_models
from ..data.loader import get_dataloaders
from .utils import get_device

def handler_compare(cfg):
    device = get_device()
    
    train_loader, test_loader, valid_loader, class_names, features_list = get_dataloaders(
        batch_size=cfg.data.batch_size,
        valid_part=cfg.data.val_part,
        num_workers=cfg.data.num_workers,
        raw_path=cfg.data.raw_dir,
        sampling_rate=cfg.data.sampling_rate,
        reduced_dataset=cfg.data.reduced_dataset,
        features=cfg.data.features
    )
    
    base_model = hydra.utils.instantiate(cfg.base_model, out_classes=len(class_names))
    handcrafted_model = None
    is_handcrafted = False
    if hasattr(cfg, "handcrafted_model"):
        handcrafted_model = hydra.utils.instantiate(cfg.handcrafted_model, base_model=base_model)
        is_handcrafted = True

    save_path = os.path.join(cfg.training.save_path, cfg.training.save_name)
    save_handcrafted_name = "handcrafted_" + cfg.training.save_name
    save_handcrafted_path = os.path.join(cfg.training.save_path, save_handcrafted_name)
    os.makedirs(cfg.training.save_path, exist_ok=True)

    if is_handcrafted and handcrafted_model is not None:
        if os.path.exists(save_handcrafted_path):
            handcrafted_model.load_state_dict(torch.load(save_handcrafted_path, map_location=device))
            print("HANDCRAFTED MODEL IS LOADED")
        else:
            print(f"NO MODEL AT {save_handcrafted_path}")
            return
    if os.path.exists(save_path):
        base_model.load_state_dict(torch.load(save_path, map_location=device))
        print("BASE MODEL IS LOADED")
    else:
        print(f"NO MODEL AT {save_path}")
        return

    model_list = [(base_model, "BASE")]
    if is_handcrafted and handcrafted_model is not None:
        model_list.append((handcrafted_model, "HANDCRAFTED"))

    for model, name in model_list:
        all_preds, all_true = evaluate_model(model, test_loader, is_handcrafted=(name == "HANDCRAFTED"))
        scores = basic_scores(all_true, all_preds)
        print(f"\n\n-----------------{name} SCORES-----------------\n")
        for score_name, score_value in scores.items():
            print(f"{score_name}: {score_value}")

    if is_handcrafted and handcrafted_model is not None:
        intervals = compare_models(base_model, handcrafted_model, test_loader)
        print("\n\n--------------------COMPARISON INTERVALS--------------------\n")
        for name, interval in intervals.items():
            print(f"{name}: ({interval[0]:.4f}, {interval[1]:.4f})")

def handler_train(cfg):
    device = get_device()
    
    train_loader, test_loader, valid_loader, class_names, features_list = get_dataloaders(
        batch_size=cfg.data.batch_size,
        valid_part=cfg.data.val_part,
        num_workers=cfg.data.num_workers,
        raw_path=cfg.data.raw_dir,
        sampling_rate=cfg.data.sampling_rate,
        reduced_dataset=cfg.data.reduced_dataset,
        features=cfg.data.features
    )
    
    model = hydra.utils.instantiate(cfg.base_model, out_classes=len(class_names))
    is_handcrafted = False
    handcrafted = None
    if hasattr(cfg, "handcrafted_model"):
        handcrafted = hydra.utils.instantiate(cfg.handcrafted_model, base_model=model)
        is_handcrafted = True

    if is_handcrafted:
        save_name = "handcrafted_" + cfg.training.save_name
        save_path = os.path.join(cfg.training.save_path, save_name)
    else:
        save_name = cfg.training.save_name
        save_path = os.path.join(cfg.training.save_path, save_name)

    os.makedirs(cfg.training.save_path, exist_ok=True)
    train_model(
        handcrafted if is_handcrafted else model,
        train_loader, test_loader, valid_loader, class_names,
        is_handcrafted=is_handcrafted,
        epochs=cfg.training.epochs,
        batch_size=cfg.data.batch_size,
        learning_rate=cfg.training.lr,
        save_path=cfg.training.save_path,
        save_name=save_name,
        features=cfg.data.features
    )
    print(f"Модель сохранена по пути: {save_path}")

def handler_evaluate(cfg):
    device = get_device()
    
    train_loader, test_loader, valid_loader, class_names, features_list = get_dataloaders(
        batch_size=cfg.data.batch_size,
        valid_part=cfg.data.val_part,
        num_workers=cfg.data.num_workers,
        raw_path=cfg.data.raw_dir,
        sampling_rate=cfg.data.sampling_rate,
        reduced_dataset=cfg.data.reduced_dataset,
        features=cfg.data.features
    )
    
    model = hydra.utils.instantiate(cfg.base_model, out_classes=len(class_names))
    is_handcrafted = False
    if hasattr(cfg, "handcrafted_model"):
        model = hydra.utils.instantiate(cfg.handcrafted_model, base_model=model)
        is_handcrafted = True

    save_name = "handcrafted_" + cfg.training.save_name if is_handcrafted else cfg.training.save_name
    save_path = os.path.join(cfg.training.save_path, save_name)

    checkpoint = torch.load(save_path, map_location=device)
    model.load_state_dict(checkpoint)
    print(f"MODEL IS LOADED FROM {save_path}")

    all_preds, all_true = evaluate_model(model, test_loader, is_handcrafted=is_handcrafted)
    scores = basic_scores(all_true, all_preds)

    print("\n\n--------------------STATISTICS--------------------\n")
    for name, score in scores.items():
        print(f"{name}: {score}")
