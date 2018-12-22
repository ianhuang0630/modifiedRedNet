import os
import argparse
import torch
import imageio
import json
import skimage.transform
import torchvision
from tqdm import tqdm

import torch.optim
import RedNet_model
from utils import utils
from utils.utils import load_ckpt

parser = argparse.ArgumentParser(description='RedNet Indoor Sementic Segmentation')

parser.add_argument('--path_rgb_test', 
                    default='data/gibson_data/img_dir_test.txt', 
                    help='path to rgb testfile')
parser.add_argument('--path_depth_test',
                    default='data/gibson_data/depth_dir_test.txt',
                    help='path to depth testfile')
parser.add_argument('--path_labels_test',
                    default='data/gibson_data/label_test.txt',
                    help='path to labels testfile')
parser.add_argument('--path_metadata_train',
                    default='data/gibson_data/meta_train.json',
                    help='path to training metadata json')
parser.add_argument('--path_metadata_test',
                    default='data/gibson_data/meta_test.json',
                    help='path to test metadata json')


# parser.add_argument('-r', '--rgb', default=None, metavar='DIR',
#                     help='path to image')
# parser.add_argument('-d', '--depth', default=None, metavar='DIR',
#                     help='path to depth')
parser.add_argument('-o', '--output', 
                    default='predictions/', metavar='DIR',
                    help='path to output directory')
parser.add_argument('--cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--last-ckpt', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

args = parser.parse_args()
device = torch.device("cuda:0" if args.cuda and torch.cuda.is_available() else "cpu")


def inference():
    
    print('loading metadata')
    if os.path.exists(args.path_metadata_train):
        meta_train = json.load(open(args.path_metadata_train))
        # load number of classes
        num_classes = meta_train['num_classes']
        image_w = meta_train['width'] 
        image_h = meta_train['height']
        colours = meta_train['colours']
    else:
        raise IOError ("could not find metadata parameters. Make sure to create the file before training.")


    # model = RedNet_model.RedNet(num_classes=44, pretrained=False)
    model = RedNet_model.RedNet(num_classes=num_classes, pretrained=False)
    load_ckpt(model, None, args.last_ckpt, device)
    model.eval()
    model.to(device)
    
    rgb_test = []
    depth_test = []
    labels_test = []
    with open(args.path_rgb_test) as f:
        lines = f.readlines()
        rgb_test = lines
    with open(args.path_depth_test) as f:
        lines = f.readlines()
        depth_test = lines
    with open(args.path_labels_test) as f: 
        lines = f.readlines()
        labels_test = lines
    assert len(rgb_test) == len(depth_test), \
            'rgb and depth lists should be the same length'
    assert len(labels_test) == len(depth_test), \
            'labels and depth lists should be the same length'
    if not os.path.exists(args.output):
        # make the folder
        os.makedirs(args.output)

    for i in tqdm(range(len(rgb_test))):
        # iterate through every image in a directory
        this_rgb = rgb_test[i].strip()
        this_depth = depth_test[i].strip()
        this_label = labels_test[i].strip()

        image = imageio.imread(this_rgb)
        depth = imageio.imread(this_depth)

        # Bi-linear
        image = skimage.transform.resize(image, (image_h, image_w), order=1,
                                         mode='reflect', preserve_range=True)
        # Nearest-neighbor
        depth = skimage.transform.resize(depth, (image_h, image_w), order=0,
                                         mode='reflect', preserve_range=True)

        image = image / 255
        image = torch.from_numpy(image).float()
        depth = torch.from_numpy(depth).float()
        image = image.permute(2, 0, 1)
        depth.unsqueeze_(0)

        image = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])(image)
        depth = torchvision.transforms.Normalize(mean=[19050],
                                                 std=[9650])(depth)

        image = image.to(device).unsqueeze_(0)
        depth = depth.to(device).unsqueeze_(0)

        pred = model(image, depth)
        # TODO: input dataset-unique color_labels
        output = utils.color_label(torch.max(pred, 1)[1] + 1)[0]
        
        base = os.path.basename(this_label)
        base = base[:-4] + '.png'
        final_output = os.path.join(args.output, base)
        
        imageio.imsave(final_output, output.cpu().numpy().transpose((1, 2, 0)))

if __name__ == '__main__':
    inference()
