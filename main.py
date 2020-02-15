import os
import sys
import argparse
import logging
from configs.parser import Parser

import torch

from apex import amp
from apex.parallel import DistributedDataParallel as ApexDDP

from model import deeplabv3, PointHead, PointRend
from datas import get_loader
from train import train
from utils.gpus import synchronize, is_main_process


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument("config", type=str, help="It must be config/*.yaml")
    parser.add_argument("save", type=str, help="Save path in out directory")
    parser.add_argument("--local_rank", type=int, default=0, help="Using for Apex DDP")
    return parser.parse_args()


def amp_init(args):
    # Apex Initialize
    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    torch.backends.cudnn.benchmark = True


def set_loggging(save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='[%y/%m/%d %H:%M:%S]')

    fh = logging.FileHandler(f"{save_dir}/log.txt")
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)


if __name__ == "__main__":
    args = parse_args()
    amp_init(args)

    parser = Parser(args.config)
    C = parser.C
    save_dir = f"{os.getcwd()}/outs/{args.save}"

    if is_main_process():
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, mode=0o775)

        parser.dump(f"{save_dir}/config.yaml")

        set_loggging(save_dir)

    device = torch.device("cuda")
    train_loader = get_loader(C.data, "train", distributed=args.distributed)
    valid_loader = get_loader(C.data, "val", distributed=args.distributed)

    net = PointRend(
        deeplabv3(**C.net.deeplab),
        PointHead(**C.net.pointhead)
    ).to(device)

    params = [{"params": net.backbone.backbone.parameters(),   "lr": C.train.lr},
              {"params": net.head.parameters(),                "lr": C.train.lr},
              {"params": net.backbone.classifier.parameters(), "lr": C.train.lr * 10}]

    optim = torch.optim.SGD(params, momentum=C.train.momentum, weight_decay=C.train.weight_decay)

    net, optim = amp.initialize(net, optim, opt_level=C.apex.opt)
    if args.distributed:
        net = ApexDDP(net, delay_allreduce=True)

    train(C.run, save_dir, train_loader, valid_loader, net, optim, device)
