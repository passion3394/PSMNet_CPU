import torch.utils.data as data

from PIL import Image
import os
import os.path
import numpy as np

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def dataloader(filepath):

  left_fold  = 'image0/'
  right_fold = 'image1/'

  #add by haoshuang
  #establish the corresponding relationship of the left image and the right image
  align_filename = filepath+'/align.txt'
  align_fp = open(align_filename,'r')
  all_lines = align_fp.readlines()
  corres = {}
  for line in all_lines:
      line_cont = line.split(',')
      if line_cont[2] != '' and line_cont[3] != '':
          corres[line_cont[2]] = line_cont[3]
  ##

  image = [img for img in os.listdir(filepath+left_fold) ]

  left_test  = [filepath+left_fold+img for img in image if img in corres and corres[img] != '']
  right_test = [filepath+right_fold+corres[img] for img in image if img in corres and corres[img] != '']

  print('the num of left:',len(left_test))
  print('the num of right:',len(right_test))

  return left_test, right_test
