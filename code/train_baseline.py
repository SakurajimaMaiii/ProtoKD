"""
train unimodal baseline
"""

import sys
import argparse
import logging
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from networks import VNet
from datasets import BraTS
from loss import DiceCeLoss
from evaluate import eval_one_dice
from evaluate import test_single_case
from utils import create_if_not


# for BraTS
CROP_SIZE = (96, 128, 128)
STRIDE = tuple([x // 2 for x in list(CROP_SIZE)])


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default="0", help="GPU to use")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument(
        "--deterministic",
        type=int,
        default=1,
        help="whether use deterministic training",
    )
    parser.add_argument("--batch_size", type=int, default=4, help="batch_size")
    parser.add_argument("--num_cls", type=int, default=4, help="the number of class")
    parser.add_argument("--num_channels", type=int, default=1, help="input channels")
    parser.add_argument(
        "--max_epoch", type=int, default=1000, help="maximum epoch number to train"
    )
    parser.add_argument("--log_dir", type=str, default="../log/lastexp", help="log dir")
    parser.add_argument("--data_dir", type=str, default="../data", help="dataset path")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument(
        "--modality",
        type=int,
        default=0,
        help="choose modality to train:0,1,2,3 for T1/T2/T1ce/Falir",
    )
    # parser.add_argument('--modality', nargs='+', type=int, default=0, help='test modality contain images ROIs to test one of [0,1,2,3]')
    parser.add_argument("--resume", type=bool, default=False, help="resume or not")
    parser.add_argument("--ckpt_path", type=str, default="", help="checkpoint path")
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    num_cls = args.num_cls
    num_channels = args.num_channels
    max_epoch = args.max_epoch
    batch_size = args.batch_size
    m = args.modality

    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    print("---Make logger")
    snapshot_path = args.log_dir
    create_if_not(snapshot_path)
    save_model_path = snapshot_path + "/model"
    create_if_not(save_model_path)

    logging.basicConfig(
        filename=snapshot_path + "/log.txt",
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    print("--Load model")
    # model
    model = VNet(
        n_channels=num_channels,
        n_classes=num_cls,
        n_filters=16,
        normalization="batchnorm",
    )
    model.train()
    model.cuda()
    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.ckpt_path)
        start_epoch = ckpt["epoch"] + 1
        model.load_state_dict(ckpt["state_dict"])
        optimizer.load_state_dict(ckpt["optimizer"])
        best_dice = ckpt["best_dice"]
        best_epoch = ckpt["best_epoch"]
    # dataset
    train_dataset = BraTS(args.data_dir, crop_size=CROP_SIZE)
    print("Training set includes %d data." % len(train_dataset))
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    imglist = []
    f = open(args.data_dir + "/val_list.txt", "r")
    lines = f.readlines()
    for ll in lines:
        imglist.append(ll.replace("\n", ""))
    f.close()
    imglist = [args.data_dir + "/brats2018/" + x + ".npy" for x in imglist]
    print("Val set includes %d data." % len(imglist))
    val_list = imglist
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    writer = SummaryWriter(snapshot_path + "/tensorboard")
    iter_num = 0
    loss_criterion = DiceCeLoss(num_cls)

    best_epoch = 0
    best_dice = 0
    best_wt = 0
    best_co = 0
    best_ec = 0
    train_time1 = time.time()

    print("---Start training.")
    for epoch in range(start_epoch, max_epoch):
        time1 = time.time()
        # change lr
        curr_lr = args.lr * (1.0 - np.float32(epoch) / np.float32(args.max_epoch)) ** (
            0.9
        )
        for parm in optimizer.param_groups:
            parm["lr"] = curr_lr
        for idx, sampled_batch in enumerate(train_loader):
            image, label = sampled_batch
            image = image[:, m : m + 1]
            image, label = image.float().cuda(), label.cuda()

            _, logits = model(image)

            dice_loss, ce_loss, loss = loss_criterion(logits, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # record loss
            iter_num = iter_num + 1
            writer.add_scalar("loss/loss", loss, iter_num)
            writer.add_scalar("loss/ce_loss", ce_loss, iter_num)
            writer.add_scalar("loss/dice_loss", dice_loss, iter_num)

            logging.info(
                "Epoch:[%d/%d],iteration:%d, loss: %f"
                % (epoch, max_epoch, iter_num, loss.item())
            )
        time2 = time.time()
        logging.info(
            "Epoch %d training time :%f minutes" % (epoch, (time2 - time1) / 60)
        )
        # val
        if epoch % 25 == 0:
            state = {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_dice": best_dice,
                "best_epoch": best_epoch,
            }
            torch.save(state, save_model_path + "/checkpoint_{}.pth.tar".format(epoch))
        # if epoch is smaller than 250, do not val, save time

        if epoch < max_epoch // 4:
            continue

        model.eval()
        dice_all_wt = []
        dice_all_co = []
        dice_all_ec = []
        dice_all_mean = []
        print("---Start epoch %d validation" % epoch)
        time1 = time.time()
        with torch.no_grad():
            for idx, val_path in enumerate(val_list):
                data = np.load(val_path)
                image = data[0:4]
                image = image[m : m + 1]
                label = data[4]

                predict, _ = test_single_case(model, image, STRIDE, CROP_SIZE, num_cls)
                dice_wt, dice_co, dice_ec, dice_mean = eval_one_dice(predict, label)
                dice_all_wt.append(dice_wt)
                dice_all_co.append(dice_co)
                dice_all_ec.append(dice_ec)
                dice_all_mean.append(dice_mean)
                logging.info("Sample [%d], average dice : %f" % (idx, dice_mean))
        time2 = time.time()
        logging.info(
            "Epoch %d validation time : %f minutes" % (epoch, (time2 - time1) / 60)
        )
        dice_all_wt = np.mean(np.array(dice_all_wt))
        dice_all_co = np.mean(np.array(dice_all_co))
        dice_all_ec = np.mean(np.array(dice_all_ec))
        dice_all_mean = np.mean(np.array(dice_all_mean))
        logging.info(
            "epoch %d val dice, wt_dice:%f, co_dice:%f, ec_dice:%f"
            % (epoch, dice_all_wt, dice_all_co, dice_all_ec)
        )
        writer.add_scalar("val/dice_wt", dice_all_wt, epoch)
        writer.add_scalar("val/dice_co", dice_all_co, epoch)
        writer.add_scalar("val/dice_ec", dice_all_ec, epoch)
        writer.add_scalar("val/dice_mean", dice_all_mean, epoch)
        if dice_all_mean >= best_dice:
            best_epoch = epoch
            best_dice = dice_all_mean
            best_wt = dice_all_wt
            best_co = dice_all_co
            best_ec = dice_all_ec
            torch.save(model.state_dict(), save_model_path + "/best_model.pth")
        model.train()
        logging.info("Best dice is: %f" % best_dice)
        logging.info("Best epoch is: %d" % best_epoch)
    writer.close()
    train_time2 = time.time()
    training_time = (train_time2 - train_time1) / 3600
    logging.info("Training finished, tensorboardX writer closed")
    logging.info("Best epoch is %d, best mean dice is %f" % (best_epoch, best_dice))
    logging.info("Dice of wt/co/ec is %f,%f,%f" % (best_wt, best_co, best_ec))
    logging.info("Training total time: %f hours." % training_time)


if __name__ == "__main__":
    main()
