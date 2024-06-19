import os
import sys
import shutil
import argparse
import logging
import time
import random
import numpy as np
import itertools
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from networks import VNet
from datasets import BraTS
from loss import DiceCeLoss, softmax_kl_loss, prototype_loss
from evaluate import eval_one_dice, test_single_case
from utils import create_if_not


CROP_SIZE = (96, 128, 128)
STRIDE = tuple([x // 2 for x in list(CROP_SIZE)])


class ProtoKD(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        print("------Configs------")
        for k, v in cfg.items():
            print(k, v)
        print("------------")

        self.num_cls = cfg["num_classes"]
        self.lr = cfg["lr"]
        # modality
        self.m = cfg["modality"]
        # temp for KD
        self.T = cfg["T"]
        # weight
        self.kd_weight = cfg["kd_weight"]
        self.proto_weight = cfg["proto_weight"]
        self.max_epoch = cfg["max_epoch"]

        # model
        self.model = VNet(
            n_channels=cfg["student_channels"],
            n_classes=self.num_cls,
            n_filters=cfg["n_filters"],
            normalization="batchnorm",
        )
        self.model.train()
        self.model.cuda()
        self.teacher_model = VNet(
            n_channels=cfg["teacher_channels"],
            n_classes=self.num_cls,
            n_filters=cfg["n_filters"],
            normalization="batchnorm",
        )
        self.teacher_model.cuda()
        self.teacher_model.load_state_dict(torch.load(cfg["teachermodel_path"]))
        self.teacher_model.eval()
        # optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=cfg["weight_decay"]
        )
        # dataloader
        train_dataset = BraTS(cfg["data_dir"], crop_size=CROP_SIZE)
        print("Training set includes %d data." % len(train_dataset))
        self.train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=cfg["batch_size"],
            shuffle=True,
            num_workers=cfg["num_workers"],
            pin_memory=True,
        )
        imglist = []
        f = open(cfg["data_dir"] + "/val_list.txt", "r")
        lines = f.readlines()
        for ll in lines:
            imglist.append(ll.replace("\n", ""))
        f.close()
        self.val_list = [cfg["data_dir"] + "/brats2018/" + x + ".npy" for x in imglist]
        print("Val set includes %d data." % len(self.val_list))

        # loss
        self.dice_ce_loss = DiceCeLoss(self.num_cls)

        # logging
        snapshot_path = cfg["log_dir"]
        create_if_not(snapshot_path)
        self.save_model_path = snapshot_path + "/model"
        create_if_not(self.save_model_path)
        logging.basicConfig(
            filename=snapshot_path + "/log.txt",
            level=logging.INFO,
            format="[%(asctime)s.%(msecs)03d] %(message)s",
            datefmt="%H:%M:%S",
        )
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
        self.writer = SummaryWriter(snapshot_path + "/tensorboard")

        self.iter_num = 0
        self.start_epoch = 0
        self.best_epoch = 0
        self.best_dice = 0
        self.best_wt = 0
        self.best_co = 0
        self.best_ec = 0

        if cfg["resume"]:
            logging.info("Load model from %s" % cfg["ckpt_path"])
            ckpt = torch.load(cfg["ckpt_path"])
            self.start_epoch = ckpt["epoch"] + 1
            self.model.load_state_dict(ckpt["state_dict"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.best_dice = ckpt["best_dice"]
        logging.info("Trainer prepared.")

    def do_one_epoch(self, epoch):
        time1 = time.time()
        for idx, sampled_batch in enumerate(self.train_loader):
            image, label = sampled_batch
            image, label = image.float().cuda(), label.cuda()
            image1 = image[:, self.m : self.m + 1]

            feature, logits = self.forward(image1)
            with torch.no_grad():
                feature_t, logits_t = self.teacher_model(image)
            # seg loss
            dice_loss, ce_loss, seg_loss = self.dice_ce_loss(logits, label)
            # kd loss
            kd_loss = softmax_kl_loss(logits / self.T, logits_t / self.T).mean()
            # proto loss
            sim_map_s, sim_map_t, proto_loss = prototype_loss(
                feature, feature_t, label, self.num_cls
            )
            # loss
            loss = seg_loss + self.kd_weight * kd_loss + self.proto_weight * proto_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.iter_num = self.iter_num + 1
            dis_feature = torch.mean((feature - feature_t) ** 2)
            dis_logits = torch.mean((logits - logits_t) ** 2)
            self.writer.add_scalar("loss/loss", loss, self.iter_num)
            self.writer.add_scalar("loss/ce_loss", ce_loss, self.iter_num)
            self.writer.add_scalar("loss/dice_loss", dice_loss, self.iter_num)
            self.writer.add_scalar("loss/seg_loss", seg_loss, self.iter_num)
            self.writer.add_scalar("loss/kd_loss", kd_loss, self.iter_num)
            self.writer.add_scalar("loss/proto_loss", proto_loss, self.iter_num)
            self.writer.add_scalar("distance/dis_feature", dis_feature, self.iter_num)
            self.writer.add_scalar("distance/dis_logits", dis_logits, self.iter_num)
            logging.info(
                "Epoch:[%d/%d],iteration:%d, loss: %f"
                % (epoch, self.max_epoch, self.iter_num, loss.item())
            )
        time2 = time.time()
        logging.info(
            "Epoch %d training time :%f minutes" % (epoch, (time2 - time1) / 60)
        )

    def validate(self, epoch):
        self.model.eval()
        dice_all_wt = []
        dice_all_co = []
        dice_all_ec = []
        dice_all_mean = []
        with torch.no_grad():
            for idx, val_path in enumerate(self.val_list):
                data = np.load(val_path)
                image = data[0:4]
                image = image[self.m : self.m + 1]
                label = data[4]

                predict, _ = test_single_case(
                    self.model, image, STRIDE, CROP_SIZE, self.num_cls
                )
                dice_wt, dice_co, dice_ec, dice_mean = eval_one_dice(predict, label)
                dice_all_wt.append(dice_wt)
                dice_all_co.append(dice_co)
                dice_all_ec.append(dice_ec)
                dice_all_mean.append(dice_mean)
                logging.info("Sample [%d], average dice : %f" % (idx, dice_mean))
        dice_all_wt = np.mean(np.array(dice_all_wt))
        dice_all_co = np.mean(np.array(dice_all_co))
        dice_all_ec = np.mean(np.array(dice_all_ec))
        dice_all_mean = np.mean(np.array(dice_all_mean))
        logging.info(
            "Epoch %d val dice, wt_dice:%f, co_dice:%f, ec_dice:%f"
            % (epoch, dice_all_wt, dice_all_co, dice_all_ec)
        )
        self.writer.add_scalar("val/dice_wt", dice_all_wt, epoch)
        self.writer.add_scalar("val/dice_co", dice_all_co, epoch)
        self.writer.add_scalar("val/dice_ec", dice_all_ec, epoch)
        self.writer.add_scalar("val/dice_mean", dice_all_mean, epoch)
        if dice_all_mean >= self.best_dice:
            self.best_epoch = epoch
            self.best_dice = dice_all_mean
            self.best_wt = dice_all_wt
            self.best_co = dice_all_co
            self.best_ec = dice_all_ec
            torch.save(
                self.model.state_dict(), self.save_model_path + "/best_model.pth"
            )
        logging.info("Epoch:%d, Best dice is %f" % (epoch, self.best_dice))
        self.model.train()

    def do_train(self):
        train_start = time.time()
        for epoch in range(self.start_epoch, self.max_epoch):
            curr_lr = self.lr * (
                1.0 - np.float32(epoch) / np.float32(self.max_epoch)
            ) ** (0.9)
            self.do_one_epoch(epoch)
            if epoch % 25 == 0:
                state = {
                    "epoch": epoch,
                    "state_dict": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "best_dice": self.best_dice,
                }
                torch.save(
                    state, self.save_model_path + "/checkpoint_{}.pth.tar".format(epoch)
                )
            if epoch < self.max_epoch // 4:
                continue
            self.validate(epoch)
        train_end = time.time()
        training_time = (train_end - train_start) / 3600
        self.writer.close()
        logging.info("Training finished, tensorboardX writer closed")
        logging.info(
            "Best epoch is %d, best mean dice is %f" % (self.best_epoch, self.best_dice)
        )
        logging.info(
            "Dice of wt/co/ec is %f,%f,%f" % (self.best_wt, self.best_co, self.best_ec)
        )
        logging.info("Training total time: %f hours." % training_time)

    def forward(self, x):
        feature, logits = self.model(x)
        return feature, logits
