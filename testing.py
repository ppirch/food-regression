import logging
from argparse import ArgumentParser
from glob import glob

from foodnet import ClassifyNet, RegressNet, SharedNet, ConcatNet
from utils.datasets import FoodImagesDataset

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import classification_report as cls_report
from sklearn.metrics import mean_absolute_percentage_error as mape

logger = logging.getLogger(__name__)


def add_model_args(parser: ArgumentParser):
    """
    Adds model arguments to the parser

    Args:
        parser (ArgumentParser): _description_
    """
    parser.add_argument(
        "--model_type",
        default="classify",
        choices=["classify", "regress", "shared", "concat"],
        help="model type",
    )
    parser.add_argument(
        "--backbone",
        default="resnet50",
        choices=["resnet50", "resnet101", "resnet152"],
        help="model backbone",
    )


def add_training_args(parser: ArgumentParser):
    """
    Adds training arguments to the parser

    Args:
        parser (ArgumentParser): _description_
    """
    parser.add_argument(
        "--loss_type",
        default="default",
        choices=["default", "uncertainty", "automatic"],
        help="loss type",
    )
    parser.add_argument(
        "--seed",
        default=677,
        type=int,
        help="seed",
    )
    parser.add_argument(
        "--batch_size",
        default=32,
        type=int,
        help="batch size",
    )
    parser.add_argument(
        "--epochs",
        default=10,
        type=int,
        help="epochs",
    )


def main():
    parser = ArgumentParser()
    add_model_args(parser)
    add_training_args(parser)
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info("Args: %s", args)

    if args.model_type in ["shared", "concat"] and args.loss_type == "default":
        raise ValueError(
            "Invalid loss type. Default is not supported for shared/concat"
        )
    if args.model_type in ["classify", "regress"] and args.loss_type != "default":
        raise ValueError(
            "Invalid loss type. Only default is supported for classify/regress"
        )

    torch.manual_seed(args.seed)

    logger.info("Loading dataset")
    DATA_DIR = "./regression"
    df = pd.read_csv(f"{DATA_DIR}/csv/train.csv")
    le = LabelEncoder()
    le.fit(df["food"])
    n_classes = len(le.classes_)

    if args.model_type == "classify":
        model = ClassifyNet(backbone=args.backbone, n_classes=n_classes)
    elif args.model_type == "regress":
        model = RegressNet(backbone=args.backbone)
    elif args.model_type == "shared":
        model = SharedNet(backbone=args.backbone, n_classes=n_classes)
    elif args.model_type == "concat":
        model = ConcatNet(backbone=args.backbone, n_classes=n_classes)
    else:
        raise ValueError("Invalid model type")

    # logger.info("MODEL: %s", model)

    test_df = pd.read_csv(f"{DATA_DIR}/csv/test.csv")
    test_datasets = FoodImagesDataset(
        csv_file=f"{DATA_DIR}/csv/test.csv",
        img_dir=f"{DATA_DIR}/test",
        target_transform=le.transform,
    )

    test_dataloader = DataLoader(
        test_datasets, batch_size=args.batch_size, shuffle=False
    )

    model_weights = glob(
        f"./models/{args.model_type}_{args.backbone}_{args.loss_type}_*_{args.seed}_.pth",  # noqa
    )
    model_weights = sorted(model_weights, key=lambda x: int(x.split("_")[-2]))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info("DEVICE: %s", device)

    for model_weight in tqdm(model_weights):
        actual_weights = []
        actual_classes = []
        predict_weights = []
        predict_classes = []

        model.load_state_dict(torch.load(model_weight))
        model.to(device)
        model.eval()

        with torch.no_grad():
            for i, data in enumerate(test_dataloader, 0):
                inputs, weight, labels = data
                actual_weights.extend(weight.flatten().numpy())
                actual_classes.extend(labels.flatten().numpy())
                weight, labels = weight.long(), labels.long()
                inputs, weight, labels = (
                    inputs.to(device, non_blocking=True),
                    weight.to(device, non_blocking=True),
                    labels.to(device, non_blocking=True),
                )
                outputs = model.forward(inputs)

                if args.model_type == "classify":
                    cls_outs = torch.argmax(outputs, 1)
                    cls_outs = cls_outs.detach().cpu().numpy()
                    predict_classes.extend(cls_outs)
                elif args.model_type == "regress":
                    reg_outs = outputs.squeeze_(1)
                    reg_outs = reg_outs.detach().cpu().numpy()
                    predict_weights.extend(reg_outs)
                elif args.model_type == "shared" or args.model_type == "concat":
                    reg_outs, cls_outs = outputs
                    reg_outs = reg_outs.squeeze_(1)
                    reg_outs = reg_outs.detach().cpu().numpy()
                    cls_outs = torch.argmax(cls_outs, 1)
                    cls_outs = cls_outs.detach().cpu().numpy()
                    predict_weights.extend(reg_outs)
                    predict_classes.extend(cls_outs)
                else:
                    raise ValueError("Invalid model type")

        if args.model_type == "classify":
            predict_weights = ["-"] * len(test_df)
        elif args.model_type == "regress":
            predict_classes = ["-"] * len(test_df)

        logger.info("Finished Predicting")
        predicts = pd.DataFrame(
            {
                "filename": test_df["filename"],
                "actual_weight": actual_weights,
                "actual_class": actual_classes,
                "predict_weight": predict_weights,
                "predict_class": predict_classes,
            }
        )
        predicts.to_csv(
            f"./predicts/{model_weight.split('/')[-1].split('.')[0]}.csv",  # noqa
            index=False,
        )

        logger.info("Model: %s", model_weight)
        if args.model_type == "classify":
            logger.info(
                "Classification report: \n %s",
                cls_report(actual_classes, predict_classes, zero_division=0),
            )
        elif args.model_type == "regress":
            logger.info(
                "MAPE: %s",
                mape(actual_weights, predict_weights),
            )
        elif args.model_type == "shared" or args.model_type == "concat":
            logger.info(
                "Classification report: \n %s",
                cls_report(actual_classes, predict_classes, zero_division=0),
            )
            logger.info(
                "MAPE: %s",
                mape(actual_weights, predict_weights),
            )


if __name__ == "__main__":
    main()
