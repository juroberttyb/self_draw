from __future__ import absolute_import, division, print_function

import os
import cv2
import numpy as np

import torch
from torch.utils.data import DataLoader

from layers import disp_to_depth
from utils import readlines
from options import MonodepthOptions
import datasets
import networks
import random
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from torchvision import transforms
from PIL import Image

from torch.utils.data import Dataset
import cv2
import numpy as np

import torch.nn.functional as nnf

import torchvision.transforms.functional as F
from torchvision.transforms.functional import convert_image_dtype

class car_seg():
    def __init__(self, opt):
        self.opt = opt

        import torchvision
        # from torchvision.models.segmentation import fcn_resnet50, fcn_resnet101, deeplabv3_resnet50, deeplabv3_resnet101
        self.model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True, progress=False)
        # self.model = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(pretrained=True, progress=False)
        # self.model = torchvision.models.segmentation.lraspp_mobilenet_v3_large(pretrained=True, progress=False)

        sem_classes = [
            '__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
            'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
            'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
        ]
        sem_class_to_idx = {cls: idx for (idx, cls) in enumerate(sem_classes)}

        import numpy as np
        self.fn = lambda x: 1. if x == sem_class_to_idx['car'] else 0. # not in [sem_class_to_idx['aeroplane'], sem_class_to_idx['bicycle'], sem_class_to_idx['bird'], sem_class_to_idx['boat'], sem_class_to_idx['bus'], sem_class_to_idx['car'], sem_class_to_idx['cat'], sem_class_to_idx['dog'], sem_class_to_idx['horse'], sem_class_to_idx['motorbike'], sem_class_to_idx['person'], sem_class_to_idx['sheep'], sem_class_to_idx['train']] else 0.
        self.fn = np.vectorize(self.fn)

        self.model.cuda()
        self.model.eval()

        self.kernel = np.ones((5,5), np.uint8)

    def __call__(self, img, ero_itr):
        "expected input of shape 3xHxW"
        # print("seg img.shape", img.shape)
        img = torch.moveaxis(img, -1, -2)
        # print("seg img.shape", img.shape)

        # width, height = img.shape[-2], img.shape[-1]
        # img = nnf.interpolate(img, size=(img.shape[-2]//2, img.shape[-1]//2), mode='bicubic', align_corners=False)

        batch = convert_image_dtype(img, dtype=torch.float)
        # batch = nnf.interpolate(batch, size=(img.shape[-2]//2, img.shape[-1]//2), mode='bilinear', align_corners=False)
        # print("batch.shape", batch.shape)

        normalized_batch = F.normalize(batch, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        output = self.model(normalized_batch.cuda())['out']

        normalized_masks = torch.nn.functional.softmax(output, dim=1)

        classes = normalized_masks.argmax(dim=1)
        weight_mask = self.fn(classes.cpu().numpy())[0]
        # print(weight_mask.mean())
        # weight_mask = nnf.interpolate(weight_mask, size=(img.shape[-2], img.shape[-1]), mode='bicubic', align_corners=False)

        weight_mask = cv2.erode(weight_mask, self.kernel, iterations=ero_itr)
        M = np.float32([[1, 0, 0], [0, 1, ero_itr]])
        rows, cols = weight_mask.shape
        weight_mask = cv2.warpAffine(weight_mask, M, (cols, rows))

        # weight_mask = np.resize(weight_mask, (img.shape[-2], img.shape[-1]))
        car_mask = weight_mask > 0.

        car_mask = np.moveaxis(car_mask, -1, -2)

        return car_mask

class WaymoDataset(Dataset):
    def __init__(self, opt):
        self.opt = opt

        self.image_path = '/image'
        self.lidar_path = '/lidar'
        self.valid = []

        import glob
        filenames = glob.glob('%s/%s/*.png' % (opt.data_path, self.image_path))
        self.file_num = len(filenames)

        '''
        self.kitti_loader = datasets.MaskedKITTIRAWDataset(opt.data_path, filenames,
                                           opt.input_size[1], opt.input_size[0],
                                           [0], 4, ero_itr=opt.ero_iter, is_train=False)
        '''

        with open('image.txt') as openfileobject:
            for line in openfileobject:
                line = int(line[:-1])
                for i in range(line, line+10):
                    self.valid.append(i)

        # print(self.valid)
        print("number of valid files: %d" % len(self.valid))

    def __len__(self):
        return self.file_num

    def transform(self, x):
        x = cv2.resize(x, (self.opt.input_size[1], self.opt.input_size[0]))

        """"add erosion and affine"""

        # print(x.shape)
        x = np.moveaxis(x, -1, 0)
        '''
        do_color_aug = False and random.random() > 0.5

        if do_color_aug:
            color_aug = transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)

        x = self.kitti_loader.preprocess(inputs=x, color_aug=color_aug)
        '''
        # print(x.shape)
        return x

    def __getitem__(self, idx):
        if idx in self.valid:
            image = Image.open(self.opt.data_path + self.image_path + '/%d.png' % (idx))
            image = np.array(image)
            image = np.moveaxis(image, 0, 1)
            # print("image.shape", image.shape)

            depth_input_image = self.transform(image)
            # lidar (x, y, depth)
            gt_depth = np.zeros(image.shape[:2])
            # print("gt_depth.shape", gt_depth.shape)

            lidar = np.load(self.opt.data_path + self.lidar_path + '/%d.npy' % (idx))
            # print("lidar.shape", lidar.shape)

            for i in range(lidar.shape[1]):
                gt_depth[int(lidar[0][i]), int(lidar[1][i])] = lidar[2][i]
            # print("gt_depth.mean()", gt_depth.mean())
            # print("gt_depth.shape", gt_depth.shape)
            # print("gt_depth", type(gt_depth))
            
            # print(lidar[2][0])
            # print(gt_depth[int(lidar[0][0]), int(lidar[1][0])])

            '''
            if self.target_transform:
                lidar = self.target_transform(lidar)
            '''

            return {'image':image, 'depth_input_image':depth_input_image, 'gt_depth':gt_depth, 'idx':idx, 'flag':True}
        return {'idx':idx, 'flag':False}

def show(imgs, n, path, fix, axs, f):
    labels = [['image', 'pred_disp'], ['gt_depth', 'car_mask'], ['mask']]

    # fix, axs = plt.subplots(ncols=2, nrows=3, squeeze=False, figsize=(32,24))
    axs[0, 0].imshow(imgs[0])
    axs[0, 0].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    axs[0, 0].set_title(label=labels[0][0])

    axs[0, 1].imshow(imgs[1])
    axs[0, 1].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    axs[0, 1].set_title(label=labels[0][1])

    tmp = cv2.resize(imgs[2], (imgs[2].shape[1]//2, imgs[2].shape[0]//2))
    ret, tmp = cv2.threshold(tmp, 0., 255., cv2.THRESH_BINARY)
    axs[1, 0].imshow(tmp, vmin=0, vmax=255) # , interpolation='nearest' # , cmap='gist_gray')
    axs[1, 0].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    axs[1, 0].set_title(label=labels[1][0])

    axs[1, 1].imshow(imgs[3])
    axs[1, 1].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    axs[1, 1].set_title(label=labels[1][1])

    tmp = f(imgs[4])
    print("non zero count:", np.count_nonzero(tmp))
    tmp = cv2.resize(tmp, (tmp.shape[1]//2, tmp.shape[0]//2))
    ret, tmp = cv2.threshold(tmp, 0., 255., cv2.THRESH_BINARY)
    axs[2, 0].imshow(tmp, vmin=0, vmax=255)
    axs[2, 0].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    axs[2, 0].set_title(label=labels[2][0])

    fix.savefig(path + '/%d.png'%(n))

# cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)

splits_dir = os.path.join(os.path.dirname(__file__), "splits")

# Models which were trained with stereo supervision were trained with a nominal
# baseline of 0.1 units. The KITTI rig has a baseline of 54cm. Therefore,
# to convert our stereo predictions to real-world scale we multiply our depths by 5.4.
STEREO_SCALE_FACTOR = 5.4

def compute_errors(gt, pred, abs_err):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    mrse = (gt - pred) ** 2
    mrse = np.sqrt(mrse)
    abs_err.append(mrse)
    mrse = mrse.mean()
    # np.save(path + "/rmse_" + str(rmse) + ".npy", np.sqrt(tmp))
    # rmse = (rmse * car_mask).sum() / car_mask.sum()

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, mrse, rmse_log, a1, a2, a3

def batch_post_process_disparity(l_disp, r_disp):
    """Apply the disparity post-processing method as introduced in Monodepthv1
    """
    _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp

def evaluate(opt):
    """Evaluates a pretrained model using a specified test set"""
    car_seg_model = car_seg(opt)
    fix, axs = plt.subplots(ncols=2, nrows=3, squeeze=False, figsize=(32,24))
    errors, ratios, abs_err, weight = [], [], [], []

    f = lambda x: 1. if x else 0.
    f = np.vectorize(f)

    assert sum((opt.eval_mono, opt.eval_stereo)) == 1, \
        "Please choose mono or stereo evaluation by setting either --eval_mono or --eval_stereo"

    if opt.eval_stereo:
        print("   Stereo evaluation - "
              "disabling median scaling, scaling by {}".format(STEREO_SCALE_FACTOR))
        opt.disable_median_scaling = True
        opt.pred_depth_scale_factor = STEREO_SCALE_FACTOR
    else:
        print("   Mono evaluation - using median scaling")

    if opt.ext_disp_to_eval is None:
        opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)

        assert os.path.isdir(opt.load_weights_folder), \
            "Cannot find a folder at {}".format(opt.load_weights_folder)

        print("-> Loading weights from {}".format(opt.load_weights_folder))

        encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
        decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")

        encoder_dict = torch.load(encoder_path)

        encoder = networks.ResnetEncoder(opt.num_layers, False)
        depth_decoder = networks.DepthDecoder(encoder.num_ch_enc)

        model_dict = encoder.state_dict()
        encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
        depth_decoder.load_state_dict(torch.load(decoder_path))

        encoder.cuda()
        encoder.eval()
        depth_decoder.cuda()
        depth_decoder.eval()

        # print("going data_path %s" % (opt.data_path))
        wayno_dataset = WaymoDataset(opt)
        dataloader = DataLoader(wayno_dataset, batch_size=1, shuffle=False, num_workers=0, # opt.num_workers,
                                pin_memory=True, drop_last=False)

        with torch.no_grad():
            for data in dataloader:
                if data['flag']:
                    image = data['image']
                    image_width, image_height = image.shape[1], image.shape[2]
                    # print("image_width, image_height", image_width, image_height)

                    depth_input_image = data['depth_input_image'].cuda()

                    car_img = torch.moveaxis(image, -1, 1)
                    car_mask = car_seg_model(car_img.cuda(), ero_itr=0) # opt.ero_iter
                    pred_disp = depth_decoder(encoder(depth_input_image))

                    pred_disp, _ = disp_to_depth(pred_disp[("disp", 0)], opt.min_depth, opt.max_depth)
                    pred_disp = pred_disp.cpu()[:, 0].numpy()
                    pred_disp = np.moveaxis(pred_disp, 0, -1)
                    pred_disp = cv2.resize(pred_disp, (image.shape[2], image.shape[1]))
                    pred_depth = 1 / pred_disp

                    gt_depth = data['gt_depth'].numpy()[0]
                    mask = np.logical_and(gt_depth > opt.MIN_DEPTH, gt_depth < opt.MAX_DEPTH)

                    crop = np.array([0.03594771 * image_width,  0.96405229 * image_width,
                                    0.40810811 * image_height, 0.99189189 * image_height]).astype(np.int32)
                    crop_mask = np.zeros(mask.shape)
                    crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
                    mask = np.logical_and(mask, crop_mask)
                    mask = np.logical_and(mask, car_mask)

                    show([np.moveaxis(image[0].numpy(), 0, 1), np.moveaxis(pred_disp, 0, 1), np.moveaxis(gt_depth, 0, 1), np.moveaxis(car_mask, 0, 1), np.moveaxis(mask, 0, 1)], data['idx'], path=opt.save_path, fix=fix, axs=axs, f=f)

                    pred_depth = pred_depth[mask]
                    gt_depth = gt_depth[mask]

                    if len(gt_depth) > 0:
                        pred_depth *= opt.pred_depth_scale_factor
                        if not opt.disable_median_scaling:
                            ratio = np.median(gt_depth) / np.median(pred_depth)
                            ratios.append(ratio)
                            pred_depth *= ratio

                        pred_depth[pred_depth < opt.MIN_DEPTH] = opt.MIN_DEPTH
                        pred_depth[pred_depth > opt.MAX_DEPTH] = opt.MAX_DEPTH

                        if pred_depth.shape[0] != 0:
                            errors.append(compute_errors(gt_depth, pred_depth, abs_err))
                            weight.append(len(gt_depth))

    if not opt.disable_median_scaling:
        ratios = np.array(ratios)
        med = np.median(ratios)
        print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))

    errors = np.array(errors)
    weight = np.array([[e] for e in weight])
    errors *= weight
    mean_errors = errors.sum(axis=0) / weight.sum()

    for i in range(len(abs_err)):
        np.save(opt.save_path + "/%d.npy" % (i), abs_err[i])

    print("num pixels captured: %d" % (weight.sum()))
    print("abs_error: %.3f" % (mean_errors[2]))

    #print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "abs_error", "rmse_log", "a1", "a2", "a3"))
    #print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
    #print("\n-> Done!")

def limit_cpu():
    from multiprocessing import Pool, cpu_count
    import psutil
    import os

    "is called at every process start"
    p = psutil.Process(os.getpid())
    p.nice(19) # 1

if __name__ == "__main__":
    # limit_cpu()
    options = MonodepthOptions().parse()
    evaluate(options)

