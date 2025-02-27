# System libs
import os
import argparse
from distutils.version import LooseVersion
from multiprocessing import Queue, Process
# Numerical libs
import numpy as np
import math
import torch
import torch.nn as nn
from scipy.io import loadmat
# Our libs
from dataset import ValDataset, SemData
from models import ModelBuilder, SegmentationModule
from models.pspnet import PSPNet
from utils import AverageMeter, colorEncode, accuracy, intersectionAndUnion, parse_devices, colorize
from lib.nn import user_scattered_collate, async_copy_to
from lib.utils import as_numpy
import lib.utils.data as torchdata
import cv2
from tqdm import tqdm
import torch.nn.functional as F
import transform
import torch.backends.cudnn as cudnn
import timeit
os.environ["CUDA_VISIBLE_DEVICES"]='0,1'

colors = loadmat('data/color11.mat')['col']


def visualize_result(data, pred, args):
    (img, seg, info) = data

    # segmentation
    seg_color = colorEncode(seg, colors)

    # prediction
    pred_color = colorEncode(pred, colors)


    im_vis = np.concatenate((seg_color, pred_color),
                             axis=1).astype(np.uint8)

    img_name = info.split('/')[-1]
    cv2.imwrite(os.path.join(args.result,
                img_name.replace('.jpg', '.png')), im_vis)





def net_process(net_encoder, net_decoder, pre_image_1, pre_image_2, pre_image_3, pre_image_5, cur_image, mean, std=None, flip=True):
    pre_input_1 = torch.from_numpy(pre_image_1.transpose((2, 0, 1))).float()
    pre_input_2 = torch.from_numpy(pre_image_2.transpose((2, 0, 1))).float()
    pre_input_3 = torch.from_numpy(pre_image_3.transpose((2, 0, 1))).float()
    pre_input_5 = torch.from_numpy(pre_image_5.transpose((2, 0, 1))).float()
    cur_input = torch.from_numpy(cur_image.transpose((2, 0, 1))).float()
    if std is None:
        for t1, m1 in zip(pre_input_1, mean):
            t1.sub_(m1)
        for t2, m2 in zip(pre_input_2, mean):
            t2.sub_(m2)
        for t3, m3 in zip(pre_input_3, mean):
            t3.sub_(m3)
        for t5, m5 in zip(pre_input_5, mean):
            t5.sub_(m5)
        for t4, m4 in zip(cur_input, mean):
            t4.sub_(m4)
    else:
        for t1, m1, s1 in zip(pre_input_1, mean, std):
            t1.sub_(m1).div_(s1)
        for t2, m2, s2 in zip(pre_input_2, mean, std):
            t2.sub_(m2).div_(s2)
        for t3, m3, s3 in zip(pre_input_3, mean, std):
            t3.sub_(m3).div_(s3)
        for t5, m5, s5 in zip(pre_input_5, mean, std):
            t5.sub_(m5).div_(s5)
        for t4, m4, s4 in zip(cur_input, mean, std):
            t4.sub_(m4).div_(s4)
    pre_input_1 = pre_input_1.unsqueeze(0).cuda()
    pre_input_2 = pre_input_2.unsqueeze(0).cuda()
    pre_input_3 = pre_input_3.unsqueeze(0).cuda()
    pre_input_5 = pre_input_5.unsqueeze(0).cuda()
    cur_input = cur_input.unsqueeze(0).cuda()
    if flip:
        pre_input_1 = torch.cat([pre_input_1, pre_input_1.flip(3)], 0)
        pre_input_2 = torch.cat([pre_input_2, pre_input_2.flip(3)], 0)
        pre_input_3 = torch.cat([pre_input_3, pre_input_3.flip(3)], 0)
        pre_input_5 = torch.cat([pre_input_5, pre_input_5.flip(3)], 0)
        cur_input = torch.cat([cur_input, cur_input.flip(3)], 0)
    with torch.no_grad():


        cur_feature_all = net_encoder(cur_input, return_feature_maps=True)

        cur_feature = cur_feature_all[-1]
        low_cur_feature = cur_feature_all[-2]

        pre_1_feature = net_encoder(pre_input_1, return_feature_maps=False)
        pre_2_feature = net_encoder(pre_input_2, return_feature_maps=False)
        pre_3_feature = net_encoder(pre_input_3, return_feature_maps=False)
        pre_5_feature = net_encoder(pre_input_5, return_feature_maps=False)

        enc_feature = []
        enc_feature.append(pre_5_feature)
        enc_feature.append(pre_3_feature)
        enc_feature.append(pre_2_feature)
        enc_feature.append(pre_1_feature)
        enc_feature.append(cur_feature)
        enc_feature.append(low_cur_feature)


        output = net_decoder(enc_feature)


    _, _, h_i, w_i = pre_input_1.shape
    _, _, h_o, w_o = output.shape

    if (h_o != h_i) or (w_o != w_i):
        output = F.interpolate(output, (h_i, w_i), mode='bilinear', align_corners=True)
    output = F.softmax(output, dim=1)
    if flip:
        output = (output[0] + output[1].flip(2)) / 2
    else:
        output = output[0]



    output = output.data.cpu().numpy()
    output = output.transpose(1, 2, 0)
    return output


def scale_process(net_encoder, net_decoder, pre_image_1, pre_image_2, pre_image_3, pre_image_5, cur_image, classes, crop_h, crop_w, h, w, mean, std=None, stride_rate=2/3):
    ori_h, ori_w, _ = pre_image_1.shape
    pad_h = max(crop_h - ori_h, 0)
    pad_w = max(crop_w - ori_w, 0)
    pad_h_half = int(pad_h / 2)
    pad_w_half = int(pad_w / 2)
    if pad_h > 0 or pad_w > 0:
        pre_image_1 = cv2.copyMakeBorder(pre_image_1, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half,
                                   cv2.BORDER_CONSTANT, value=mean)
        pre_image_2 = cv2.copyMakeBorder(pre_image_2, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half,
                                         cv2.BORDER_CONSTANT, value=mean)
        pre_image_3 = cv2.copyMakeBorder(pre_image_3, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half,
                                         cv2.BORDER_CONSTANT, value=mean)
        pre_image_5 = cv2.copyMakeBorder(pre_image_5, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half,
                                         cv2.BORDER_CONSTANT, value=mean)
        cur_image = cv2.copyMakeBorder(cur_image, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half,
                                       cv2.BORDER_CONSTANT, value=mean)

    new_h, new_w, _ = pre_image_1.shape
    stride_h = int(np.ceil(crop_h * stride_rate))
    stride_w = int(np.ceil(crop_w * stride_rate))
    grid_h = int(np.ceil(float(new_h - crop_h) / stride_h) + 1)
    grid_w = int(np.ceil(float(new_w - crop_w) / stride_w) + 1)
    prediction_crop = np.zeros((new_h, new_w, classes), dtype=float)
    count_crop = np.zeros((new_h, new_w), dtype=float)
    for index_h in range(0, grid_h):
        for index_w in range(0, grid_w):
            s_h = index_h * stride_h
            e_h = min(s_h + crop_h, new_h)
            s_h = e_h - crop_h
            s_w = index_w * stride_w
            e_w = min(s_w + crop_w, new_w)
            s_w = e_w - crop_w
            pre_image_1_crop = pre_image_1[s_h:e_h, s_w:e_w].copy()
            pre_image_2_crop = pre_image_2[s_h:e_h, s_w:e_w].copy()
            pre_image_3_crop = pre_image_3[s_h:e_h, s_w:e_w].copy()
            pre_image_5_crop = pre_image_5[s_h:e_h, s_w:e_w].copy()
            cur_image_crop = cur_image[s_h:e_h, s_w:e_w].copy()
            count_crop[s_h:e_h, s_w:e_w] += 1
            prediction_crop[s_h:e_h, s_w:e_w, :] += net_process(net_encoder, net_decoder, pre_image_1_crop, pre_image_2_crop, pre_image_3_crop, pre_image_5_crop, cur_image_crop, mean, std)
    prediction_crop /= np.expand_dims(count_crop, 2)
    prediction_crop = prediction_crop[pad_h_half:pad_h_half + ori_h, pad_w_half:pad_w_half + ori_w]
    prediction = cv2.resize(prediction_crop, (w, h), interpolation=cv2.INTER_LINEAR)
    return prediction



def evaluate(net_encoder, net_decoder, loader, args, result_queue):
    #segmentation_module.eval()

    net_encoder.eval()
    net_decoder.eval()


    crop_h = 360
    crop_w = 480

    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]

    j = 0

    #for i, (input, _) in enumerate(loader):
    for batch_data in loader:
        # process data
        batch_data = batch_data[0]
        seg_label = as_numpy(batch_data['seg_label'][0])
        #print(np.unique(seg_label))

        j = j + 1


        #input = batch_data['img_data']
        pre_input_1 = batch_data['pre_imgdata_1']
        pre_input_1= np.squeeze(pre_input_1.numpy(), axis=0)
        pre_image_1 = np.transpose(pre_input_1, (1, 2, 0))

        pre_input_2 = batch_data['pre_imgdata_2']
        pre_input_2= np.squeeze(pre_input_2.numpy(), axis=0)
        pre_image_2 = np.transpose(pre_input_2, (1, 2, 0))

        pre_input_3 = batch_data['pre_imgdata_3']
        pre_input_3= np.squeeze(pre_input_3.numpy(), axis=0)
        pre_image_3 = np.transpose(pre_input_3, (1, 2, 0))

        pre_input_5 = batch_data['pre_imgdata_5']
        pre_input_5= np.squeeze(pre_input_5.numpy(), axis=0)
        pre_image_5 = np.transpose(pre_input_5, (1, 2, 0))

        cur_input = batch_data['cur_imgdata']
        cur_input= np.squeeze(cur_input.numpy(), axis=0)
        cur_image = np.transpose(cur_input, (1, 2, 0))



        h, w, _ = pre_image_1.shape
        prediction = np.zeros((h, w, args.num_class), dtype=float)
        for scale in args.scales:
            long_size = round(scale * 512)
            new_h = long_size
            new_w = long_size
            if h > w:
                new_w = round(long_size / float(h) * w)
            else:
                new_h = round(long_size / float(w) * h)
            pre_image_1_scale = cv2.resize(pre_image_1, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            pre_image_2_scale = cv2.resize(pre_image_2, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            pre_image_3_scale = cv2.resize(pre_image_3, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            pre_image_5_scale = cv2.resize(pre_image_5, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            cur_image_scale = cv2.resize(cur_image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            prediction += scale_process(net_encoder, net_decoder, pre_image_1_scale, pre_image_2_scale, pre_image_3_scale, pre_image_5_scale, cur_image_scale, args.num_class, crop_h, crop_w, h, w, mean, std)
        prediction /= len(args.scales)

        prediction = np.argmax(prediction, axis=2)

        print('j:', j)


        # calculate accuracy and SEND THEM TO MASTER

        intersection, union, target = intersectionAndUnion(prediction, seg_label, args.num_class)
        result_queue.put_nowait((intersection, union, target))

        # visualization
        if args.visualize:
            visualize_result(
                (batch_data['img_ori'], seg_label, batch_data['cur_info']),
                prediction, args)




def worker(args, result_queue):
    #torch.cuda.set_device(gpu_id)

    # Dataset and Loader
    dataset_val = ValDataset(
        args.list_val, args, max_sample=args.num_val,
        start_idx=-1, end_idx=-1)
    loader_val = torchdata.DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=user_scattered_collate,
        num_workers=2)


    # Network Builders
    builder = ModelBuilder()

    net_encoder = builder.build_encoder(
        arch=args.arch_encoder,
        fc_dim=args.fc_dim,
        weights=args.weights_encoder)



    net_decoder = builder.build_decoder(
        arch=args.arch_decoder,
        fc_dim=args.fc_dim,
        num_class=args.num_class,
        weights=args.weights_decoder,
        use_softmax=True)


    crit = nn.NLLLoss(ignore_index=11)


    net_encoder = torch.nn.DataParallel(net_encoder).cuda()
    net_decoder = torch.nn.DataParallel(net_decoder).cuda()

    cudnn.benchmark = True


    # Main loop
    evaluate(net_encoder, net_decoder, loader_val, args, result_queue)


def main(args):

    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    with open(args.list_val, 'r') as f:
        lines = f.readlines()
        num_files = len(lines)
    print('num:',num_files)

    result_queue = Queue(2000)

    worker(args, result_queue)

    # master fetches results
    processed_counter = 0
    while processed_counter < num_files:
        if result_queue.empty():
            continue
        (intersection, union, target) = result_queue.get()
        #acc_meter.update(acc, pix)
        intersection_meter.update(intersection)
        union_meter.update(union)
        target_meter.update(target)
        processed_counter += 1


    # summary

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)


    for i in range(args.num_class):
        print('class [{}], IoU: {:.4f}'.format(i, iou_class[i]))


    print('[Eval Summary]:')
    print('Mean IoU: {:.4f}, mAcc: {:.4f}, allAcc: {:.4f}'
          .format(mIoU, mAcc, allAcc))

    print('Evaluation Done!')


if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), \
        'PyTorch>=0.4.0 is required'

    parser = argparse.ArgumentParser()
    # Model related arguments
    parser.add_argument('--id', required=True,
                        help="a name for identifying the model to load")
    parser.add_argument('--suffix', default='_epoch_20.pth',
                        help="which snapshot to load")

    parser.add_argument('--arch_decoder', default='nonkeyc1',
                        help="architecture of net_decoder")
    parser.add_argument('--arch_encoder', default='resnet50dilated',
                        help="architecture of net_encoder")

    parser.add_argument('--fc_dim', default=2048, type=int,
                        help='number of features between encoder and decoder')
    parser.add_argument('--scales', default=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75], type=int)

    # Path related arguments
    parser.add_argument('--list_val',
                        default='./data/testcamvid.odgt')
    parser.add_argument('--root_dataset',
                        default='./data/camvid_video/')

    # Data related arguments
    parser.add_argument('--num_val', default=-1, type=int,
                        help='number of images to evalutate')
    parser.add_argument('--num_class', default=11, type=int,
                        help='number of classes')
    parser.add_argument('--batch_size', default=1, type=int,
                        help='batchsize. current only supports 1')
    parser.add_argument('--imgSize', default=[713], nargs='+', type=int,
                        help='list of input image sizes.'
                             'for multiscale testing, e.g.  300 400 500 600')
    parser.add_argument('--imgMaxSize', default=1000, type=int,
                        help='maximum input image size of long edge')
    parser.add_argument('--padding_constant', default=8, type=int,
                        help='maxmimum downsampling rate of the network')

    # Misc arguments
    parser.add_argument('--ckpt', default='./ckpt',
                        help='folder to output checkpoints')
    parser.add_argument('--visualize', action='store_true',
                        help='output visualization?')
    parser.add_argument('--result', default='./result',
                        help='folder to output visualization results')
    parser.add_argument('--gpus', default='0',
                        help='gpu ids for evaluation')

    args = parser.parse_args()

    args.arch_encoder = args.arch_encoder.lower()
    args.arch_decoder = args.arch_decoder.lower()
    print("Input arguments:")
    for key, val in vars(args).items():
        print("{:16} {}".format(key, val))

    # absolute paths of model weights

    args.weights_teacher = os.path.join('/media/DATA3/asm/Pycharm-Project/semseg-master/exp/cityscapes/pspnet50/model/train_epoch_200.pth')

    args.weights_decoder = os.path.join(args.ckpt, args.id,
                                        'decoder' + args.suffix)
    args.weights_encoder = os.path.join(args.ckpt, args.id,
                                        'encoder' + args.suffix)

    assert os.path.exists(args.weights_decoder), 'checkpoint does not exitst!'
    assert os.path.exists(args.weights_encoder), 'checkpoint does not exitst!'

    args.result = os.path.join(args.result, args.id)
    if not os.path.isdir(args.result):
        os.makedirs(args.result)

    main(args)
