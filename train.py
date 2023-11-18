import argparse
import collections
import warnings

import numpy as np
import torch

import hw_tts.loss as module_loss
import hw_tts.metric as module_metric
import hw_tts.model as module_arch
from hw_tts.trainer import Trainer
from hw_tts.utils import prepare_device
from hw_tts.utils.object_loading import get_dataloaders
from hw_tts.utils.parse_config import ConfigParser
import hw_tts.utils.lr_scheduler


warnings.filterwarnings("ignore", category=UserWarning)

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main(config):
    logger = config.get_logger("train")

    # setup data_loader instances
    dataloaders = get_dataloaders(config)
    
    # build model architecture, then print to console
    model = config.init_obj(config["arch"], module_arch)
    logger.info(model)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config["n_gpu"], logger)
    logger.info(f"Device {device}")
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    
    # get function handles of loss and metrics
    loss_module = config.init_obj(config["loss"], module_loss).to(device)
    metrics = [
        config.init_obj(metric_dict, module_metric)
        for metric_dict in config["metrics"]
    ]

    # build optimizer, learning rate scheduler. delete every line containing lr_scheduler for
    # disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj(config["optimizer"], torch.optim, trainable_params)
    if "lr_scheduler" in config.config:
        lr_scheduler = config.init_obj(config["lr_scheduler"], hw_tts.utils.lr_scheduler, optimizer)
    else:
        lr_scheduler = None
    trainer = Trainer(
        model,
        loss_module,
        metrics,
        optimizer,
        config=config,
        device=device,
        dataloaders=dataloaders,
        lr_scheduler=lr_scheduler,
        len_epoch=config["trainer"].get("len_epoch", None)
    )

    trainer.train()


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )
    args.add_argument(
        "-k",
        "--wandb_key",
        default=None,
        type=str,
        help="WanDB API key",
    )
    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple("CustomArgs", "flags type target")
    options = [
        CustomArgs(["--lr", "--learning_rate"], type=float, target="optimizer;args;lr"),
        CustomArgs(
            ["--n_gpu"], type=int, target="n_gpu"
        ),
        CustomArgs(
            ["--bs", "--batch_size"], type=int, target="data;train;batch_size"
        ),
        CustomArgs(
            ["--batch_expand_size"], type=int, target="trainer;batch_expand_size"
        ),
        CustomArgs(
            ["--len_epoch"], type=int, target="trainer;len_epoch"
        ),
        CustomArgs(
            ["--wandb_run_name"], type=str, target="trainer;wandb_run_name"
        ),
        CustomArgs(
            ["--waveglow_path"], type=str, target="trainer;waveglow_path"
        ),
        CustomArgs(
            ["--data_path"], type=str, target="data;train;datasets;[0];args;data_path"
        ),
        CustomArgs(
            ["--mel_ground_truth"], type=str, target="data;train;datasets;[0];args;mel_ground_truth"
        ),
        CustomArgs(
            ["--alignment_path"], type=str, target="data;train;datasets;[0];args;alignment_path"
        ),
        CustomArgs(
            ["--pitch_path"], type=str, target="data;train;datasets;[0];args;pitch_path"
        ),
        CustomArgs(
            ["--energy_path"], type=str, target="data;train;datasets;[0];args;energy_path"
        ),
        CustomArgs(
            ["--limit"], type=int, target="data;train;datasets;[0];args;limit"
        )
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
