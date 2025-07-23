import os
import json
import hydra
from omegaconf import DictConfig, OmegaConf
from ..training.trainer import train_model
from ..training.evaluator import evaluate_model, basic_scores, compare_models
from ..data.loader import get_dataloaders
from .utils import get_device, load_model, save_model


def handler_compare(cfg):
    device = get_device()

    train_loader, test_loader, valid_loader, class_names, features_list = (
        get_dataloaders(
            batch_size=cfg.training.batch_size,
            num_workers=cfg.training.num_workers,
            raw_path=cfg.data.raw_dir,
            sampling_rate=cfg.data.sampling_rate,
            pathologies=cfg.data.pathologies,
            features=cfg.data.features,
        )
    )

    base_model = hydra.utils.instantiate(cfg.base_model, out_classes=len(class_names))
    handcrafted_model = None
    is_handcrafted = False
    if (
        hasattr(cfg, "handcrafted_model")
        and cfg.handcrafted_model is not None
        and hasattr(cfg.handcrafted_model, "_target_")
    ):
        handcrafted_model = hydra.utils.instantiate(
            cfg.handcrafted_model,
            base_model=base_model,
            handcrafted_classes=len(features_list),
        )
        is_handcrafted = True

    save_path = os.path.join(cfg.training.save_path, f"{cfg.training.model_name}")
    save_handcrafted_name = f"handcrafted_{cfg.training.model_name}"
    save_handcrafted_path = os.path.join(cfg.training.save_path, save_handcrafted_name)
    os.makedirs(cfg.training.save_path, exist_ok=True)

    if is_handcrafted and handcrafted_model is not None:
        if os.path.exists(save_handcrafted_path):
            load_model(handcrafted_model, save_handcrafted_path)
            print("HANDCRAFTED MODEL IS LOADED")
        else:
            raise FileNotFoundError(f"NO MODEL AT {save_handcrafted_path}")

    if os.path.exists(save_path):
        load_model(base_model, save_path)
        print("BASE MODEL IS LOADED")
    else:
        raise FileNotFoundError(f"NO MODEL AT {save_path}")

    model_list = [(base_model, "BASE")]
    if is_handcrafted and handcrafted_model is not None:
        model_list.append((handcrafted_model, "HANDCRAFTED"))

    for model, name in model_list:
        all_preds, all_true = evaluate_model(
            model, test_loader, is_handcrafted=(name == "HANDCRAFTED")
        )
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

    train_loader, test_loader, valid_loader, class_names, features_list = (
        get_dataloaders(
            batch_size=cfg.training.batch_size,
            num_workers=cfg.training.num_workers,
            raw_path=cfg.data.raw_dir,
            sampling_rate=cfg.data.sampling_rate,
            pathologies=cfg.data.pathologies,
            features=cfg.data.features,
        )
    )

    model = hydra.utils.instantiate(cfg.base_model, out_classes=len(class_names))
    is_handcrafted = False
    handcrafted = None
    if (
        hasattr(cfg, "handcrafted_model")
        and cfg.handcrafted_model is not None
        and hasattr(cfg.handcrafted_model, "_target_")
    ):
        handcrafted = hydra.utils.instantiate(
            cfg.handcrafted_model,
            base_model=model,
            handcrafted_classes=len(features_list),
        )
        is_handcrafted = True

    model_name = cfg.training.model_name

    if is_handcrafted:
        model_name = f"handcrafted_{model_name}"

    os.makedirs(cfg.training.save_path, exist_ok=True)

    _, metadata = train_model(
        handcrafted if is_handcrafted else model,
        train_loader,
        test_loader,
        valid_loader,
        class_names,
        is_handcrafted=is_handcrafted,
        learning_rate=cfg.training.lr,
        model_name=model_name,
    )

    model_save_path = os.path.join(cfg.training.save_path, f"{model_name}.pth")
    save_model(model, model_save_path)
    print(f"Модель сохранена по пути: {model_save_path}")

    json_name = f"{model_name}.json"
    json_save_path = os.path.join(cfg.training.metadata_save_path, json_name)
    with open(json_save_path, "w") as output:
        json.dump(metadata, output)
    print(f"Метаданные сохранены в файле: {json_save_path}")


def handler_evaluate(cfg):
    device = get_device()

    train_loader, test_loader, valid_loader, class_names, features_list = (
        get_dataloaders(
            batch_size=cfg.training.batch_size,
            num_workers=cfg.training.num_workers,
            raw_path=cfg.data.raw_dir,
            sampling_rate=cfg.data.sampling_rate,
            pathologies=cfg.data.pathologies,
            features=cfg.data.features,
        )
    )

    model = hydra.utils.instantiate(cfg.base_model, out_classes=len(class_names))
    is_handcrafted = False
    if (
        hasattr(cfg, "handcrafted_model")
        and cfg.handcrafted_model is not None
        and hasattr(cfg.handcrafted_model, "_target_")
    ):
        model = hydra.utils.instantiate(
            cfg.handcrafted_model,
            base_model=model,
            handcrafted_classes=len(features_list),
        )
        is_handcrafted = True

    model_name = (
        f"handcrafted_{cfg.training.model_name}"
        if is_handcrafted
        else cfg.training.model_name
    )
    save_path = os.path.join(cfg.training.save_path, f"{model_name}.pth")
    try:
        load_model(model, save_path)
        print(f"MODEL IS LOADED FROM {save_path}")
    except:
        raise FileNotFoundError(f"NO MODEL AT {save_path}")

    all_preds, all_true = evaluate_model(
        model, test_loader, is_handcrafted=is_handcrafted
    )
    scores = basic_scores(all_true, all_preds)

    print("\n\n--------------------STATISTICS--------------------\n")
    for name, score in scores.items():
        print(f"{name}: {score}")
