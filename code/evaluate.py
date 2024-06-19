import math

import torch
import torch.nn.functional as F
import numpy as np
import medpy.metric.binary as mmb
import SimpleITK as sitk


def test_single_case(net, image, stride, patch_size, num_classes=1):
    """
    predict 3d volume using slide window

    Parameters
    ----------
        net : model
        image : must be 3d array,shape [C,W,H,D]
        stride : tuple / List
        patch_size : tuple / List
        num_classes : number of class

    Returns
    -------
    label_map : prediction, shape is the same as image
    score_map : softmax outputs, shape [C,*]
    """
    _, w, h, d = image.shape

    # if the size of image is less than patch_size, then padding it
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0] - w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1] - h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2] - d
        add_pad = True
    else:
        d_pad = 0
    wl_pad, wr_pad = w_pad // 2, w_pad - w_pad // 2
    hl_pad, hr_pad = h_pad // 2, h_pad - h_pad // 2
    dl_pad, dr_pad = d_pad // 2, d_pad - d_pad // 2
    if add_pad:
        image = np.pad(
            image,
            [(0, 0), (wl_pad, wr_pad), (hl_pad, hr_pad), (dl_pad, dr_pad)],
            mode="constant",
            constant_values=0,
        )
    _, ww, hh, dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride[0]) + 1
    sy = math.ceil((hh - patch_size[1]) / stride[1]) + 1
    sz = math.ceil((dd - patch_size[2]) / stride[2]) + 1
    score_map = np.zeros((num_classes,) + (ww, hh, dd)).astype(np.float32)
    cnt = np.zeros((ww, hh, dd)).astype(np.float32)

    for x in range(0, sx):
        xs = min(stride[0] * x, ww - patch_size[0])
        for y in range(0, sy):
            ys = min(stride[1] * y, hh - patch_size[1])
            for z in range(0, sz):
                zs = min(stride[2] * z, dd - patch_size[2])
                test_patch = image[
                    :,
                    xs : xs + patch_size[0],
                    ys : ys + patch_size[1],
                    zs : zs + patch_size[2],
                ]
                test_patch = np.expand_dims(test_patch, axis=0).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).cuda()
                _, y1 = net(test_patch)
                y = F.softmax(y1, dim=1)
                y = y.cpu().data.numpy()
                y = y[0, :, :, :, :]
                score_map[
                    :,
                    xs : xs + patch_size[0],
                    ys : ys + patch_size[1],
                    zs : zs + patch_size[2],
                ] = (
                    score_map[
                        :,
                        xs : xs + patch_size[0],
                        ys : ys + patch_size[1],
                        zs : zs + patch_size[2],
                    ]
                    + y
                )
                cnt[
                    xs : xs + patch_size[0],
                    ys : ys + patch_size[1],
                    zs : zs + patch_size[2],
                ] = (
                    cnt[
                        xs : xs + patch_size[0],
                        ys : ys + patch_size[1],
                        zs : zs + patch_size[2],
                    ]
                    + 1
                )
    score_map = score_map / np.expand_dims(cnt, axis=0)
    label_map = np.argmax(score_map, axis=0)
    if add_pad:
        label_map = label_map[
            wl_pad : wl_pad + w, hl_pad : hl_pad + h, dl_pad : dl_pad + d
        ]
        score_map = score_map[
            :, wl_pad : wl_pad + w, hl_pad : hl_pad + h, dl_pad : dl_pad + d
        ]
    return label_map, score_map


def decode_label(label):
    """
    Convert multi-label to region label
    label, 1(ET),2(NET),3(ED)
    """
    label = label.copy()
    wt = (label != 0) * 1
    tc = (label == 1) * 1 + (label == 2) * 1
    ec = (label == 1) * 1
    return wt, tc, ec


def eval_dice(pred, label):
    """
    dice to eval
    """
    if np.sum(pred) == 0 and np.sum(label) == 0:
        return 1
    else:
        return mmb.dc(pred, label)


def eval_one_dice(pred, label):
    """
    for validation
    """
    pred_data_wt, pred_data_co, pred_data_ec = decode_label(pred)
    gt_data_wt, gt_data_co, gt_data_ec = decode_label(label)

    dice_wt = eval_dice(pred_data_wt, gt_data_wt)
    dice_co = eval_dice(pred_data_co, gt_data_co)
    dice_ec = eval_dice(pred_data_ec, gt_data_ec)
    dice_mean = (dice_wt + dice_co + dice_ec) / 3.0

    return dice_wt, dice_co, dice_ec, dice_mean


def evaluate_one_case(pred, label):
    """evaluate one case
    metric : dice hd sensitivity specificity
    dice: (2*TP)/(FP+2*TP+FN)
    sensitivity : TP/(TP+FN)
    specificity : TN/(TN+FP)
    """
    pred_data_wt, pred_data_co, pred_data_ec = decode_label(pred)
    gt_data_wt, gt_data_co, gt_data_ec = decode_label(label)

    if np.sum(pred_data_wt) > 0 and np.sum(gt_data_wt) > 0:
        hd_wt = mmb.hd95(pred_data_wt, gt_data_wt)
    else:
        hd_wt = np.nan

    if np.sum(pred_data_co) > 0 and np.sum(gt_data_co) > 0:
        hd_co = mmb.hd95(pred_data_co, gt_data_co)
    else:
        hd_co = np.nan

    if np.sum(pred_data_ec) > 0 and np.sum(gt_data_ec) > 0:
        hd_ec = mmb.hd95(pred_data_ec, gt_data_ec)
    else:
        hd_ec = np.nan

    hd = [hd_wt, hd_co, hd_ec]

    dice_wt = eval_dice(pred_data_wt, gt_data_wt)
    dice_co = eval_dice(pred_data_co, gt_data_co)
    dice_ec = eval_dice(pred_data_ec, gt_data_ec)

    dice = [dice_wt, dice_co, dice_ec]

    sensitivity_wt = mmb.sensitivity(pred_data_wt, gt_data_wt)
    sensitivity_co = mmb.sensitivity(pred_data_co, gt_data_co)
    sensitivity_ec = mmb.sensitivity(pred_data_ec, gt_data_ec)

    sensitivity = [sensitivity_wt, sensitivity_co, sensitivity_ec]

    specificity_wt = mmb.specificity(pred_data_wt, gt_data_wt)
    specificity_co = mmb.specificity(pred_data_co, gt_data_co)
    specificity_ec = mmb.specificity(pred_data_ec, gt_data_ec)

    specificity = [specificity_wt, specificity_co, specificity_ec]

    return hd, dice, sensitivity, specificity


def convert_to_sitk(arr, output_path, modality=None):
    if modality is not None:
        for i in modality:
            itkimg = sitk.GetImageFromArray(arr[i])
            sitk.WriteImage(itkimg, output_path + "/image_%d.nii.gz" % i)
    else:
        itkimg = sitk.GetImageFromArray(arr)
        sitk.WriteImage(itkimg, output_path)


if __name__ == "__main__":

    import os
    import logging
    import sys
    import argparse
    import shutil

    from tqdm import tqdm

    from networks import VNet

    """
    Evaluate on test set.
    aLL results would be organized as follows:
    output_path
       -- vis
          --0
            image.nii.gz
            label.nii.gz
            predict.nii.gz
          --1
          --2
       --results.txt
       --reults.npy
    """
    parser = argparse.ArgumentParser("Evaluate on test set")
    parser.add_argument("--gpu", type=str, default="0", help="GPU to use")
    parser.add_argument(
        "--num_channels", type=int, default=1, help="the number of input channels"
    )
    parser.add_argument("--data_dir", type=str, default="../data", help="dataset path")
    parser.add_argument(
        "--model_path",
        type=str,
        default="../log/0409baseline/model/epoch_880.pth",
        help="test model path",
    )
    parser.add_argument("--num_cls", type=int, default=4, help="the number of class")
    parser.add_argument(
        "--test_modality",
        nargs="+",
        type=int,
        default=1,
        help="test modality contain images ROIs to test one of [0,1,2,3]",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="../results/PMKL_T1",
        help="results save path",
    )
    parser.add_argument(
        "--save_vis", action="store_true", help="save vis results or not"
    )
    args = parser.parse_args()
    dic = vars(args)
    for k, v in dic.items():
        print(k, v)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    num_channels = args.num_channels
    num_cls = args.num_cls
    model_path = args.model_path
    data_dir = args.data_dir
    CROP_SIZE = (96, 128, 128)
    STRIDE = tuple([x // 2 for x in list(CROP_SIZE)])
    test_modality = args.test_modality
    output_path = args.output_path

    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print("create path: %s" % output_path)

    print("### Load trained model")
    model = VNet(
        n_channels=num_channels,
        n_classes=num_cls,
        n_filters=16,
        normalization="batchnorm",
    )
    model.cuda()
    model.load_state_dict(torch.load(model_path))
    shutil.copy(model_path, output_path + "/model.pth")
    model.eval()

    print("### Prepare test dataset")
    imglist = []
    f = open(data_dir + "/test_list.txt", "r")
    lines = f.readlines()
    for line in lines:
        imglist.append(line.replace("\n", ""))
    f.close()
    imglist = [data_dir + "/brats2018/" + x + ".npy" for x in imglist]
    print("### test set has %d volumes" % len(imglist))

    dice_arr = []
    hd_arr = []
    sen_arr = []
    spe_arr = []

    # test and write to log
    logging.basicConfig(
        filename=output_path + "/results.txt",
        level=logging.INFO,
        format="%(message)s",
        datefmt="%S",
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    print("------------------------------------------------------------------------")
    with torch.no_grad():
        for idx, data_path in enumerate(tqdm(imglist, ncols=70, colour="#9999FF")):
            name_vol = data_path.split("/")[-1]
            name_vol = name_vol.split(".")[0]
            data = np.load(data_path)
            image = data[0:4]
            label = data[4]
            imgs = []
            # get input data teacher or student
            if len(test_modality) == 1:
                imgs = image[test_modality[0]]
                imgs = np.expand_dims(imgs, axis=0).astype(np.float32)
            else:
                for d in test_modality:
                    imgs.append(image[d])
                imgs = np.stack(imgs)
            assert len(test_modality) == num_channels, "input channels must match!"

            predict, _ = test_single_case(model, imgs, STRIDE, CROP_SIZE, num_cls)

            if args.save_vis:
                os.makedirs(output_path + "/%s" % name_vol)
                convert_to_sitk(
                    image, output_path + "/%s" % name_vol, modality=test_modality
                )
                convert_to_sitk(
                    label.astype(np.uint8),
                    output_path + "/%s" % name_vol + "/label.nii.gz",
                )
                convert_to_sitk(
                    predict.astype(np.uint8),
                    output_path + "/%s" % name_vol + "/predict.nii.gz",
                )
            hd, dice, sen, spe = evaluate_one_case(predict, label)
            dice_arr.append(dice)
            hd_arr.append(hd)
            sen_arr.append(sen)
            spe_arr.append(spe)
            dice_mean = np.mean(np.array(dice))
            logging.info("%s  average dice is %f." % (name_vol, dice_mean))

    hd_arr = np.array(hd_arr)
    dice_arr = np.array(dice_arr) * 100
    sen_arr = np.array(sen_arr)
    spe_arr = np.array(spe_arr)

    dice_mean = np.nanmean(dice_arr, 0)
    hd_mean = np.nanmean(hd_arr, 0)
    sen_mean = np.nanmean(sen_arr, 0)
    spe_mean = np.nanmean(spe_arr, 0)

    np.save(output_path + "/hd_arr.npy", hd_arr)
    np.save(output_path + "/dice_arr.npy", dice_arr)
    np.save(output_path + "/spe_arr.npy", spe_arr)
    np.save(output_path + "/sen_arr.npy", sen_arr)

    logging.info("Statistical indicators on test set(wt/co/ec):")
    logging.info("Dice:[%.2f,%.2f,%.2f]" % (dice_mean[0], dice_mean[1], dice_mean[2]))
    logging.info("HD:[%.5f,%.5f,%.5f]" % (hd_mean[0], hd_mean[1], hd_mean[2]))
    logging.info("Sen:[%.5f,%.5f,%.5f]" % (sen_mean[0], sen_mean[1], sen_mean[2]))
    logging.info("Spe:[%.5f,%.5f,%.5f]" % (spe_mean[0], spe_mean[1], spe_mean[2]))

    logging.info(
        "Average dice is {}.".format((dice_mean[0] + dice_mean[1] + dice_mean[2]) / 3.0)
    )
    logging.info(
        "Average hd is {}.".format((hd_mean[0] + hd_mean[1] + hd_mean[2]) / 3.0)
    )
    logging.info(
        "Average spe is {}.".format((spe_mean[0] + spe_mean[1] + spe_mean[2]) / 3.0)
    )
    logging.info(
        "Average sen is {}.".format((sen_mean[0] + sen_mean[1] + sen_mean[2]) / 3.0)
    )
    print("Evaluation is finished")
    print("#####################")
