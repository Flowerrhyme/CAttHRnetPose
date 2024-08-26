import torch
from torch.utils.data import Dataset
import os 
from pathlib import Path
import glob
from tqdm import tqdm
import numpy as np
from PIL import Image
import cv2
from common.general import xywhn2xyxy, xyxy2xywh

img_formats = ['jpg', 'jpeg', 'png'] 

def img2label_paths(img_paths):
    # Define label paths as a function of image paths
    sa, sb = os.sep + 'radar_images' + os.sep, os.sep + 'labels' + os.sep  # /images/, /labels/ substrings
    return ['txt'.join(x.replace(sa, sb, 1).rsplit(x.split('.')[-1], 1)) for x in img_paths]

def get_hash(files):
    # Returns a single hash value of a list of files
    return sum(os.path.getsize(f) for f in files if os.path.isfile(f))

class LoadImagesAndLabels(Dataset):  # for training/testing
    def __init__(self, path, cache_path, img_size=640, hyp=None):
        self.img_size = img_size
        self.hyp = hyp
        self.path = path
        self.flip_index = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]

        try:
            f = []  # image files
            for p in path if isinstance(path, list) else [path]:
                p = Path(p)  # os-agnostic
                if p.is_dir():  # dir
                    f += glob.glob(str(p / '**' / '*.*'), recursive=True)
                    # f = list(p.rglob('**/*.*'))  # pathlib
                else:
                    raise Exception(f'{p} does not exist')
            self.img_files = [x.replace('/', os.sep).split(' ')[0] for x in f if x.split(' ')[0].split('.')[-1].lower() in img_formats]
            sorted_index = [i[0] for i in sorted(enumerate(self.img_files), key=lambda x:x[1])]
            self.img_files = [self.img_files[index] for index in sorted_index]

            # self.img_files = sorted([x for x in f if x.suffix[1:].lower() in img_formats])  # pathlib
            assert self.img_files, f'No images found'
        except Exception as e:
            raise Exception(f'Error loading data from {path}: {e}')

        # Check cache
        self.label_files = img2label_paths(self.img_files)  # labels
        cache_path = Path(cache_path).with_suffix('.cache')
        if cache_path.is_file():
            cache, exists = torch.load(cache_path), True  # load
            if cache['hash'] != get_hash(self.label_files + self.img_files) or 'version' not in cache:  # changed
                cache, exists = self.cache_labels(cache_path), False  # re-cache
        else:
            cache, exists = self.cache_labels(cache_path), False  # cache

        # Display cache
        nf, nm, ne, nc, n = cache.pop('results')  # found, missing, empty, corrupted, total
        if exists:
            d = f"Scanning '{cache_path}' images and labels... {nf} found, {nm} missing, {ne} empty, {nc} corrupted"
            tqdm(None, desc=d, total=n, initial=n)  # display cache results
        assert nf > 0, f'No labels in {cache_path}. Can not train without labels.'

        # Read cache
        cache.pop('hash')  # remove hash
        cache.pop('version')  # remove version
        labels, shapes, self.segments = zip(*cache.values())
        self.labels = list(labels)
        self.shapes = np.array(shapes, dtype=np.float64)
        self.img_files = list(cache.keys())  # update
        self.label_files = img2label_paths(cache.keys())  # update
        
        for x in self.labels:
            x[:, 0] = 0

        n = len(shapes)  # number of images
        self.n = n
        self.indices = range(n)
            
    def __len__(self):
        return len(self.img_files)
    
    def collate_fn(batch):
        img, label, path, shapes = zip(*batch)  # transposed
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0), path, shapes
    
    def cache_labels(self, path=Path('./labels.cache')):
        # Cache dataset labels, check images and read shapes
        x = {}  # dict
        nm, nf, ne, nc = 0, 0, 0, 0  # number missing, found, empty, duplicate
        pbar = tqdm(zip(self.img_files, self.label_files), desc='Scanning images', total=len(self.img_files))
        for i, (im_file, lb_file) in enumerate(pbar):
            try:
                # verify images
                im = Image.open(im_file)
                im.verify()  # PIL verify
                shape = im.size
                segments = []  # instance segments
                assert (shape[0] > 9) & (shape[1] > 9), f'image size {shape} <10 pixels'
                assert im.format.lower() in img_formats, f'invalid image format {im.format}'

                # verify labels
                if os.path.isfile(lb_file):
                    nf += 1  # label found
                    with open(lb_file, 'r') as f:
                        l = [x.split() for x in f.read().strip().splitlines()]
                        l = np.array(l, dtype=np.float32)
                    if len(l):
                        assert (l >= 0).all(), 'negative labels'
                        assert l.shape[1] == 56, 'labels require 56 columns each'
                        assert (l[:, 5::3] <= 1).all(), 'non-normalized or out of bounds coordinate labels'
                        assert (l[:, 6::3] <= 1).all(), 'non-normalized or out of bounds coordinate labels'
                        # print("l shape", l.shape)
                        kpts = np.zeros((l.shape[0], 39))
                        for i in range(len(l)):
                            kpt = np.delete(l[i,5:], np.arange(2, l.shape[1]-5, 3))  #remove the occlusion paramater from the GT
                            kpts[i] = np.hstack((l[i, :5], kpt))
                        l = kpts
                        assert l.shape[1] == 39, 'labels require 39 columns each after removing occlusion paramater'
                       

                        assert np.unique(l, axis=0).shape[0] == l.shape[0], 'duplicate labels'
                        x[im_file] = [l, shape, segments]
                    else:
                        ne += 1  # label empty
                else:
                    nm += 1  # label missing
                
            except Exception as e:
                nc += 1
                print(f'WARNING: Ignoring corrupted image and/or label {im_file}: {e}')

            pbar.desc = f"Scanning '{path.parent / path.stem}' images and labels... " \
                        f"{nf} found, {nm} missing, {ne} empty, {nc} corrupted"
        pbar.close()

        if nf == 0:
            print(f'WARNING: No labels found in {path}.')

        x['hash'] = get_hash(self.label_files + self.img_files)
        x['results'] = nf, nm, ne, nc, i + 1
        x['version'] = 0.1  # cache version

        try:
            torch.save(x, path)  # save for next time
            print(f'cache create {path}')
            
        except Exception as e:
            print(e) 

        return x
    
    def __getitem__(self, index):
        index = self.indices[index]  # linear, shuffled, or image_weights
        
        # Load image
        img, (h0, w0), (h, w) = load_image(self, index)

        shape = self.img_size  # final letterboxed shape
        letterbox1 = letterbox(img, shape, auto=False)
        img, ratio, pad = letterbox1
        shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

        labels = self.labels[index].copy()
        if labels.size:  # normalized xywh to pixel xyxy format
            labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])

        nL = len(labels)  # number of labels
        if nL:
            labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])  # convert xyxy to xywh
            labels[:, [2, 4]] /= img.shape[0]  # normalized height 0-1
            labels[:, [1, 3]] /= img.shape[1]  # normalized width 0-1  
            labels[:, 6::2] /= img.shape[0]  # normalized kpt heights 0-1
            labels[:, 5::2] /= img.shape[1] # normalized kpt width 0-1

        num_kpts = (labels.shape[1]-5)//2
        labels_out = torch.zeros((nL, 6+2*num_kpts))
        if nL: 
            labels_out[:, 1:] = torch.from_numpy(labels)

        img = np.ascontiguousarray(img)

        return torch.from_numpy(img).permute(2,0,1), labels_out, self.img_files[index], shapes


class LoadImages(Dataset):  # for training/testing
    def __init__(self, path, cache_path, img_size=640, hyp=None):
        self.img_size = img_size
        self.hyp = hyp
        self.path = path
        self.flip_index = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]

        try:
            f = []  # image files
            for p in path if isinstance(path, list) else [path]:
                p = Path(p)  # os-agnostic
                if p.is_dir():  # dir
                    f += glob.glob(str(p / '**' / '*.*'), recursive=True)
                    # f = list(p.rglob('**/*.*'))  # pathlib
                else:
                    raise Exception(f'{p} does not exist')
            self.img_files = [x.replace('/', os.sep).split(' ')[0] for x in f if x.split(' ')[0].split('.')[-1].lower() in img_formats]
            sorted_index = [i[0] for i in sorted(enumerate(self.img_files), key=lambda x:x[1])]
            self.img_files = [self.img_files[index] for index in sorted_index]

            # self.img_files = sorted([x for x in f if x.suffix[1:].lower() in img_formats])  # pathlib
            assert self.img_files, f'No images found'
        except Exception as e:
            raise Exception(f'Error loading data from {path}: {e}')



        n = len(self.img_files)  # number of images
        self.n = n
        self.indices = range(n)
            
    def __len__(self):
        return len(self.img_files)
    
    def collate_fn(batch):
        img, path, shapes = zip(*batch)  # transposed

        return torch.stack(img, 0), path, shapes
    
   
    def __getitem__(self, index):
        index = self.indices[index]  # linear, shuffled, or image_weights
        
        # Load image
        img, (h0, w0), (h, w) = load_image(self, index)

        shape = self.img_size  # final letterboxed shape
        letterbox1 = letterbox(img, shape, auto=False)
        img, ratio, pad = letterbox1
        shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling



        img = np.ascontiguousarray(img)

        return torch.from_numpy(img).permute(2,0,1), self.img_files[index], shapes



def load_image(self, index):

    path = self.img_files[index]
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)  
    assert img is not None, 'Image Not Found ' + path
    h0, w0 = img.shape[:2]  # orig hw
    r = self.img_size / max(h0, w0)  # resize image to img_size
    if r != 1:  # always resize down, only resize up if training with augmentation
        interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
        img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
    return img, (h0, w0), img.shape[:2]  # img, hw_original, hw_resized




def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)



    
