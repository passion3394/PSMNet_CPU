from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import skimage
import skimage.io
import skimage.transform
import numpy as np
import time
import math
import warnings
import sys
import cv2

from utils import preprocess 
from models import *

# 2012 data /media/jiaren/ImageNet/data_scene_flow_2012/testing/

parser = argparse.ArgumentParser(description='PSMNet')
parser.add_argument('--KITTI', default='2015',
                    help='KITTI version')
parser.add_argument('--datapath', default='/media/jiaren/ImageNet/data_scene_flow_2015/testing/',
                    help='select model')
parser.add_argument('--result_path', default='./result_dir/', help='the dir to save the result')
parser.add_argument('--loadmodel', default=None,
                    help='loading model')
parser.add_argument('--model', default='stackhourglass',
                    help='select model')
parser.add_argument('--maxdisp', type=int, default=192,
                    help='maxium disparity')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

#kitti padding size
#padding_size_x = 384
#padding_size_y = 1248

#roadlinks padding size
#padding_size_x = 688
padding_size_x = 768
padding_size_y = 1232

#apollo padding size
#padding_size_x = 688
#padding_size_y = 1696

if args.cuda:
    torch.cuda.manual_seed(args.seed)
else:
    torch.manual_seed(args.seed)

if args.KITTI == '2015':
   from dataloader import KITTI_submission_loader as DA
else:
   from dataloader import KITTI_submission_loader2012 as DA  


test_left_img, test_right_img = DA.dataloader(args.datapath)

if args.model == 'stackhourglass':
    model = stackhourglass(args.maxdisp)
elif args.model == 'basic':
    model = basic(args.maxdisp)
else:
    print('no model')

if args.cuda:
    #added by haoshuang 20181225 for engineering
    dev_id = 0
    print(dev_id)
    if not dev_id:
        model = nn.DataParallel(model, device_ids=[0])
    else:
        dev_id = int(dev_id)
        warn_message = 'the dev_id to DataParallel is: {0}'.format(dev_id)
        warnings.warn(warn_message)
        model = nn.DataParallel(model, device_ids=[dev_id])
    model.cuda()
else:
    model = model.cpu()
    print('cpu')

if args.loadmodel is not None:
    if args.cuda:
        state_dict = torch.load(args.loadmodel)
        model.load_state_dict(state_dict['state_dict'])
    else:
        state_dict = torch.load(args.loadmodel,map_location='cpu')
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict['state_dict'].items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        # load params
        model.load_state_dict(new_state_dict)

print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

def test(imgL,imgR):
    model.eval()

    if args.cuda:
        imgL = torch.FloatTensor(imgL).cuda()
        imgR = torch.FloatTensor(imgR).cuda()     
        imgL, imgR= Variable(imgL), Variable(imgR)
    else:
        imgL = Variable(torch.from_numpy(imgL))
        imgR = Variable(torch.from_numpy(imgR))

    with torch.no_grad():
        output = model(imgL,imgR,args.cuda)
    output = torch.squeeze(output)
    pred_disp = output.data.cpu().numpy()

    return pred_disp

def main():
   processed = preprocess.get_transform(augment=False)

   for inx in range(len(test_left_img)):
       if not os.path.exists(test_left_img[inx]):
           continue
       imgL_o = cv2.imread(test_left_img[inx])
       shp = imgL_o.shape
       imgL_o = cv2.resize(imgL_o, (shp[1]//2, shp[0]//2), interpolation=cv2.INTER_LINEAR)
       imgL_o = imgL_o.astype('float32')
       if not os.path.exists(test_right_img[inx]):
           continue
       imgR_o = cv2.imread(test_right_img[inx])
       imgR_o = cv2.resize(imgR_o, (shp[1]//2, shp[0]//2), interpolation=cv2.INTER_LINEAR)
       imgR_o = imgR_o.astype('float32')

       #imgL_o = (skimage.io.imread(test_left_img[inx]).astype('float32'))
       #imgR_o = (skimage.io.imread(test_right_img[inx]).astype('float32'))
       imgL = processed(imgL_o).numpy()
       imgR = processed(imgR_o).numpy()
       imgL = np.reshape(imgL,[1,3,imgL.shape[1],imgL.shape[2]])
       imgR = np.reshape(imgR,[1,3,imgR.shape[1],imgR.shape[2]])

       top_pad = padding_size_x-imgL.shape[2]
       left_pad = padding_size_y-imgL.shape[3]
       imgL = np.lib.pad(imgL,((0,0),(0,0),(top_pad,0),(0,left_pad)),mode='constant',constant_values=0)
       imgR = np.lib.pad(imgR,((0,0),(0,0),(top_pad,0),(0,left_pad)),mode='constant',constant_values=0)

       start_time = time.time()
       pred_disp = test(imgL,imgR)
       print('time = %.2f' %(time.time() - start_time))

       top_pad   = padding_size_x-imgL_o.shape[0]
       left_pad  = padding_size_y-imgL_o.shape[1]
       img = pred_disp[top_pad:,:-left_pad]
       if not os.path.exists(args.result_path):
           os.makedirs(args.result_path)
       skimage.io.imsave(args.result_path+test_left_img[inx].split('/')[-1][:-4]+'.png',(img*256).astype('uint16'))

if __name__ == '__main__':
   main()
