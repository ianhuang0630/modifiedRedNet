import os
from scipy.misc import imread, imsave
import numpy as np
from tqdm import tqdm
import random
import json


def npy_to_depth(flist, save_loc):
    # generates a png and saves into predesignated location 
    if not os.path.exists(save_loc):
        os.mkdir(save_loc)
    img_loc = []    
    for path in tqdm(flist):
        # remove the .npy, and plug in .png
        img_filepath = os.path.basename(path)[:-3]+'png'
        img_filepath = os.path.join(save_loc, img_filepath) 
        if not os.path.exists(img_filepath):
            depth = np.load(path)
            depth = depth.reshape(depth.shape[1], depth.shape[0])
            # save image
            imsave(img_filepath, depth)
        img_loc.append(img_filepath)
    return img_loc

def npy_to_rgb(flist, save_loc):
    # generates a png and saves into predesignated location 
    if not os.path.exists(save_loc):
        os.mkdir(save_loc)
    
    img_loc = []    
    for path in tqdm(flist):
        # remove the .npy, and plug in .png
        img_filepath = os.path.basename(path)[:-3]+'png'
        img_filepath = os.path.join(save_loc, img_filepath) 
        if not os.path.exists(img_filepath):
            rgb = np.load(path)
            rgb = rgb.reshape(rgb.shape[1], rgb.shape[0], -1)
            # save image
            rgb = rgb[:,:,:3]
            imsave(img_filepath, rgb)
        img_loc.append(img_filepath)
    return img_loc


def npy_to_labels(flist, save_loc):
    if not os.path.exists(save_loc):
        os.mkdir(save_loc)
    
    img_loc = []    
    for path in tqdm(flist):
        # remove the .npy, and plug in .png
        img_filepath = os.path.basename(path)[:-3]+'npy'
        img_filepath = os.path.join(save_loc, img_filepath) 
        if not os.path.exists(img_filepath):
            labels = np.load(path)
            labels = labels[:,:,2]
            labels = labels.reshape(labels.shape[1], labels.shape[0])
            # save .npy
            np.save(img_filepath, labels) 
        img_loc.append(img_filepath)
    return img_loc

def generate_metadata(rgb_file_path, depth_filepath, labels_filepath, save_loc):
    """ Generate a json file containing the image dimensions, the number of
        classes and the med class frequency, and colours for every single class
        all saved in a json file in the desired save_location.
    Inputs:
        rgb_filepath (str): path to training rgb images
        depth_filepath (str): path to training depth images
        labels_filepath (str): path to training labels
        save_loc (str): save location path
    """
    # calculate dimensions of images
    # To maximize efficiency, the function assumes that images are of the same
    # dimension
    with open(train_rgb_filepath, mode='r') as f:
        image_path = f.readline().strip()
        trial_image = imread(image_path)
        assert isinstance(trial_image, np.ndarray), \
                'image should be loaded as np array'
        dimensions = trial_image.shape

    # number of classes
    class_count = {}
    with open(labels_file_path, mode='r') as f:
        for line in tqdm(f):
            # load the labels
            assert '.npy' in line, 'the labels should be stored in .npy'
            labels = np.load(line.strip()).flatten()
            for element in labels:
                if element not in class_count:
                    class_count[element] = 1
                else:
                    class_count[element] += 1
    total_pixel_count = np.sum(class_count.values())
    class_prob = {key: class_count[key]/float(total_pixel_count) \
                    for key in class_count}
    prob_median = np.median(class_prod.values())
    med_freq = {key: prob_median/float(class_prob[key]) for key in class_prob}
    # get num_classes
    num_classes = len(class_count)
    # generate random colours, add on the original
    colours = [(0,0,0)]
    for i in range(num_classes):
        colours.append(
            (random.randint(low, high), random.randint(low, high), 
                random.randint(low, high)))
    assert len(colours) == num_classes + 1
    
    json_dict = {'height': dimensions[0], 'width': dimensions[1], 
                'colours': colours, 'med_freq': med_freq, 
                'num_classes': len(class_count), 'class_prob': class_prob}
    
    with open(save_loc, 'w') as f:
        f.dump(json_dict, f)
    
    return json_dict


if __name__=='__main__':
    data_dir = 'data/minos_data/minos_training_data'
    rgb_dir = 'data/minos_data/rgb'
    depth_dir = 'data/minos_data/depth'
    label_dir = 'data/minos_data/label'
    
    # adopting the way done in Rednet
    train_rgb_filepath = 'data/minos_data/img_dir_train.txt'
    train_depth_filepath = 'data/minos_data/depth_dir_train.txt'
    train_labels_filepath = 'data/minos_data/label_train.txt'
    train_meta_filepath = 'data/minos_data/meta_train.json'

    test_rgb_filepath = 'data/minos_data/img_dir_test.txt'
    test_depth_filepath = 'data/minos_data/depth_dir_test.txt'
    test_labels_filepath = 'data/minos_data/label_test.txt'
    test_meta_filepath = 'data/minos_data/meta_test.json'

    rgb_npy_path = []
    depth_npy_path = []
    label_npy_path = []
    
    for f in os.listdir(data_dir):
        if 'color' in f:
            # add the filename
            rgb_npy_path.append(os.path.join(data_dir, f))
            head = f[:f.index('color')]
            tail = f[f.index('color')+len('color'):]
            depth_npy_path.append(os.path.join(data_dir, 
                                                head + 'depth' + tail))
            label_npy_path.append(os.path.join(data_dir, 
                                                head + 'object_type' + tail))

    assert len(rgb_npy_path) == len(depth_npy_path) \
            and len(depth_npy_path) == len(label_npy_path), \
            'list lengths should be the same'
    
    # saving all the npy arrays
    depth_img_path = np.array(npy_to_depth(depth_npy_path, depth_dir))
    rgb_img_path = np.array(npy_to_rgb(rgb_npy_path, rgb_dir))
    labels_img_path = np.array(npy_to_labels(label_npy_path, label_dir))
    
    # create the training test split files
    train_ratio = 0.8
    train_idx = np.random.choice(len(depth_img_path), 
                                int(train_ratio*len(depth_img_path)),
                                replace=False)
    test_idx = np.array(list(set(range(len(depth_img_path))) -
                    set (train_idx)))

    # writing into the training files
    with open(train_rgb_filepath, 'w') as f:
        paths = rgb_img_path[train_idx]
        f.writelines(["%s \n" % item for item in paths])
    with open(train_depth_filepath, 'w') as f:
        paths = depth_img_path[train_idx]
        f.writelines(["%s \n" % item for item in paths])
    with open(train_labels_filepath, 'w') as f:
        paths = labels_img_path[train_idx]
        f.writelines(["%s \n" % item for item in paths])
    
    # writing into the testing files

    with open(test_rgb_filepath, 'w') as f:
        paths = rgb_img_path[test_idx]
        f.writelines(["%s \n" % item for item in paths])
    with open(test_depth_filepath, 'w') as f:
        paths = depth_img_path[test_idx]
        f.writelines(["%s \n" % item for item in paths])
    with open(test_labels_filepath, 'w') as f:
        paths = labels_img_path[test_idx]
        f.writelines(["%s \n" % item for item in paths])
    
    # generate the metadata files
    generate_metadata(train_rgb_filepath, train_depth_filepath, 
                    train_labels_filepath, train_meta_filepath)
    generate_metadata(test_rgb_filepath, test_depth_filepath, 
                    test_labels_filepath, test_meta_filepath)

