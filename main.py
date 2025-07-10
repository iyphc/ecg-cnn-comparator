import os
import argparse
from numpy import False_
import torch
from src.models.utils import get_device
from src.models.train import train_model
from src.models.evaluation import evaluate_model
from src.data.loader import get_dataloaders
from src.models.cnn_handcrafted import HandcraftedModel
from src.models.base_model import BaseModel
from src.models.evaluation import compare_models
from src.models.evaluation import basic_scores

def parse_args():
    parser = argparse.ArgumentParser(description="Train/Test/Compare ECG Models with optional features")

    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--train", action="store_true", help="Train model")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate model")
    parser.add_argument("--compare", action="store_true", help="Compare models")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--handcrafted", action="store_true", help="Use handcrafted features")
    parser.add_argument("--statistics", action="store_true", help="Print statistics")
    return parser.parse_args()

def main():
    args = parse_args()
    train_loader, test_loader, valid_loader, class_names, features_num = get_dataloaders(batch_size=args.batch_size)
    out_classes = len(class_names)
    if args.compare:
        models = list()
        models.append(BaseModel(
            in_channels=12,
            out_classes=out_classes
        ))
        models.append(HandcraftedModel(
            in_channels=12,
            out_classes=out_classes,
            handcrafted_classes=features_num
        ))
        if args.train or not os.path.exists("handcrafted_CNN_ECG_detection.pth"):
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
            print("HANDCRAFTED MODEL IS LOADED")
        if args.train or not os.path.exists("CNN_ECG_detection.pth"):
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
            print("BASE MODEL IS LOADED")
        
        all_preds, all_true = evaluate_model(models[0], test_loader, is_handcrafted=False)
        scores = basic_scores(all_true, all_preds)
        print("\n\n-----------------")
        print("BASE MODEL SCORES")
        print("-----------------\n")
        for name in scores:
            print(f"{name}: {scores[name]}")

        all_preds, all_true = evaluate_model(models[1], test_loader, is_handcrafted=True)
        scores = basic_scores(all_true, all_preds)
        print("\n\n------------------------")
        print("HANDCRAFTED MODEL SCORES")
        print("------------------------\n")
        for name in scores:
            print(f"{name}: {scores[name]}")

        intervals = compare_models(models[0], models[1], test_loader)
        print("\n\n--------------------")
        print("COMPARISON INTERVALS")
        print("--------------------\n")
        for name in intervals:
            print(f"{name}: ({intervals[name][0]:.4f}, {intervals[name][1]:.4f})")
    elif args.train:
        model_tmp = None

        if args.handcrafted:
            model_tmp = HandcraftedModel(
                in_channels=12,
                out_classes=out_classes,
                handcrafted_classes=features_num
            )
        else:
            model_tmp = BaseModel(
                in_channels=12,
                out_classes=out_classes
            )

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
    elif args.statistics and not args.compare:
        model_tmp = None
        if args.handcrafted:
            model_tmp = HandcraftedModel(
                in_channels=12,
                out_classes=out_classes,
                handcrafted_classes=features_num
            )
        else:
            model_tmp = BaseModel(
                in_channels=12,
                out_classes=out_classes
            )

        have_to_train = True

        if (not args.handcrafted) and os.path.exists("CNN_ECG_detection.pth"):
            state_dict = torch.load("CNN_ECG_detection.pth", weights_only=False)
            model_tmp.load_state_dict(state_dict)
            have_to_train = False
            print("BASE MODEL IS LOADED\n")
        elif args.handcrafted and os.path.exists("handcrafted_CNN_ECG_detection.pth"):
            state_dict = torch.load("handcrafted_CNN_ECG_detection.pth", weights_only=False)
            model_tmp.load_state_dict(state_dict)
            have_to_train = False
            print("HANDCRAFTED MODEL IS LOADED\n")
        
        if have_to_train:
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
        all_preds, all_true = evaluate_model(model_tmp, test_loader, is_handcrafted=args.handcrafted)
        scores = basic_scores(all_true, all_preds)

        for name in scores:
            print(f"{name}: {scores[name]}")
    else:
        print("\n------\nWARNING\n------\n")
        print("PLEASE ADD FLAGS\n")


if __name__ == "__main__":
    main()
