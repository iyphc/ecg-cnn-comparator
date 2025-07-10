import os
import argparse
import torch
from src.models.utils import get_device
from src.models.train import train_model
from src.models.evaluation import evaluate_model
from src.data.loader import get_dataloaders
from src.models.cnn_handcrafted import HandcraftedModel
from src.models.base_model import BaseModel
from src.models.cnn_handcrafted import HandcraftedModel

def parse_args():
    parser = argparse.ArgumentParser(description="Train ECG Model with optional handcrafted features")

    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--train", action="store_true", help="Train model")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate model")
    parser.add_argument("--compare", action="store_true", help="Compare models")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--handcrafted", action="store_true", help="Use handcrafted features")
    return parser.parse_args()

def main():
    args = parse_args()
    print(f"[INFO] Handcrafted features: {'ENABLED' if args.handcrafted else 'DISABLED'}")

    train_loader, test_loader, valid_loader, class_names = get_dataloaders(batch_size=args.batch_size)

    out_classes = len(class_names)
    handcrafted_dim = getattr(train_loader.dataset, "handcrafted_dim", 0)
    print(handcrafted_dim)

    if args.compare:
        models = list()
        models[0] = BaseModel(
            in_channels=12,
            out_classes=out_classes
        )
        models[1] = HandcraftedModel(
            in_channels=12,
            out_classes=out_classes,
            handcrafted_classes=handcrafted_dim
        )
        if not os.path.exists("handcrafted_CNN_ECG_detection.pth"):
            models[1] = train_model(
                models[1],
                train_loader,
                test_loader,
                valid_loader,
                class_names,
                is_handcrafted=True,
            epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.lr
            )
        else:
            state_dict = torch.load("handcrafted_CNN_ECG_detection.pth", weights_only=False)
            models[1].load_state_dict(state_dict)
        if not os.path.exists("CNN_ECG_detection.pth"):
            models[0] = train_model(
                models[0],
                train_loader,
                test_loader,
                valid_loader,
                class_names,
                is_handcrafted=False,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.lr
            )
        else:
            state_dict = torch.load("CNN_ECG_detection.pth", weights_only=False)
            models[0].load_state_dict(state_dict)
        
        scores = list()
        scores[0] = evaluate_model(models[0], test_loader, False)
        scores[1] = evaluate_model(models[1], test_loader, True)

    else:
        model_tmp = None

        if args.handcrafted:
            model_tmp = HandcraftedModel(
                in_channels=12,
                out_classes=out_classes,
                handcrafted_classes=handcrafted_dim
            )
        else:
            model_tmp = BaseModel(
                in_channels=12,
                out_classes=out_classes
            )

        # Запуск обучения
        train_model(
            model_tmp,
            train_loader,
            test_loader,
            valid_loader,
            class_names,
            is_handcrafted=args.handcrafted,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr
        )

if __name__ == "__main__":
    main()
