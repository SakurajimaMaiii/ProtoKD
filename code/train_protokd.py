"""
train KD + protoKD
"""

import argparse
from trainer import ProtoKD
from utils import set_random
from config import CONFIG as cfg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument(
        "--deterministic",
        type=int,
        default=1,
        help="whether use deterministic training",
    )
    parser.add_argument(
        "--modality",
        type=int,
        default=2,
        help="choose modality to train:0,1,2,3 for T1/T2/T1ce/Falir",
    )
    parser.add_argument("--resume", type=bool, default=False, help="resume or not")
    parser.add_argument(
        "--proto_weight", type=float, default=0.1, help="proto loss weight"
    )
    parser.add_argument("--kd_weight", type=float, default=10, help="kd loss weight")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--log_dir", type=str, default="../log/lastexp", help="log dir")
    parser.add_argument("--ckpt_path", type=str, default="", help="checkpoint path")
    args = parser.parse_args()

    cfg["modality"] = args.modality
    cfg["proto_weight"] = args.proto_weight
    cfg["kd_weight"] = args.kd_weight
    cfg["log_dir"] = args.log_dir
    cfg["resume"] = args.resume
    cfg["ckpt_path"] = args.ckpt_path
    model = ProtoKD(cfg)
    model.do_train()
