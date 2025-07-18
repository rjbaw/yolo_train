import glob
import math
import os
import random
import shutil
import time
import uuid

from pathlib import Path
from threading import Thread

import codecs
from xml.etree import ElementTree
from xml.etree.ElementTree import Element, SubElement
from lxml import etree

import cv2
import numpy as np
import torch
from PIL import Image, ExifTags
from torch.utils.data import Dataset
from tqdm import tqdm

from utils.utils import xyxy2xywh, xywh2xyxy
from utils.parse_config import parse_data_cfg

img_formats = ['.bmp', '.jpg', '.jpeg', '.png', '.tiff', '.dng']
vid_formats = ['.mov', '.avi', '.mp4']
XML_EXT = '.xml'
TXT_EXT = '.txt'
ENCODE_METHOD = 'utf-8'

# Get orientation exif tag
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break

def exif_size(img):
    # Returns exif-corrected PIL size
    s = img.size  # (width, height)
    try:
        rotation = dict(img._getexif().items())[orientation]
        if rotation == 6:  # rotation 270
            s = (s[1], s[0])
        elif rotation == 8:  # rotation 90
            s = (s[1], s[0])
    except:
        pass
    return s

class rename_rand_pair:
    def __init__(self, save_dir, label_suffix, img_suffix):
        self.save_dir = save_dir
        self.label_suffix = label_suffix
        self.img_suffix = img_suffix
    def check_duplicate(self):
        names = []
        save = []
        for fn in os.listdir(self.save_dir):
            name, suffix = os.path.splitext(fn)
            if name in names:
                save.append(name)
                names.remove(name)
            else:
                names.append(name)
        return save, names
    def remove_duplicate(self, dup):
        for fn in dup:
            labelpath = self.con_label(fn)
            if os.path.exists(labelpath):
                os.remove(labelpath)
            for suffix in self.img_suffix:
                imgpath = self.con_img(fn, suffix)
                if os.path.exists(imgpath):
                    os.remove(imgpath)
    def get_suffix(self, name):
        suffix = [os.path.splitext(n)[-1] \
                  for n in os.listdir(self.save_dir) \
                  if (os.path.splitext(n)[0]==name)] 
        for suf in suffix:
            if suf != self.label_suffix:
                return suf
    def rename_pair(self, names):
        for fn in names:
            name = uuid.uuid4().hex
            labelpath = self.con_label(name)
            while os.path.exists(labelpath):
                name = uuid.uuid4().hex
                labelpath = self.con_label(name)
            os.rename(self.con_label(fn), labelpath)
            suffix = self.get_suffix(fn)
            imgpath = self.con_img(name, suffix)
            os.rename(self.con_img(fn, suffix), imgpath)
    def main(self):
        names, dup = self.check_duplicate()
        self.remove_duplicate(dup)
        self.rename_pair(names)
    def con_label(self, name):
        return os.path.join(self.save_dir, name + self.label_suffix)
    def con_img(self, name, suffix):
        return os.path.join(self.save_dir, name + suffix)

class PascalVocReader:
    def __init__(self, filepath):
        # shapes type:
        # [label, [(x1,y1), (x2,y2), (x3,y3), (x4,y4)], color, color, difficult]
        self.shapes = []
        self.imagedata = []
        self.filepath = filepath
        self.verified = False
        try:
            self.parseXML()
        except:
            pass
    def getShapes(self):
        return self.shapes
    def addShape(self, label, bndbox, difficult):
        xmin = int(float(bndbox.find('xmin').text))
        ymin = int(float(bndbox.find('ymin').text))
        xmax = int(float(bndbox.find('xmax').text))
        ymax = int(float(bndbox.find('ymax').text))
        points = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
        self.shapes.append((label, points, None, None, difficult))
    def addImageData(self, size):
        img_width = int(float(size.find('width').text))
        img_height = int(float(size.find('height').text))
        depth = int(float(size.find('depth').text))
        self.imagedata = [img_height, img_width, depth]
#        imageShape = [image.height(), image.width(), 1 if image.isGrayscale() else 3]
    def getImageData(self):
        return self.imagedata
    def parseXML(self):
        assert self.filepath.endswith(XML_EXT), "Unsupport file format"
        parser = etree.XMLParser(encoding=ENCODE_METHOD)
        xmltree = ElementTree.parse(self.filepath, parser=parser).getroot()
        filename = xmltree.find('filename').text
        size = xmltree.find("size")
        self.addImageData(size)
        try:
            verified = xmltree.attrib['verified']
            if verified == 'yes':
                self.verified = True
        except KeyError:
            self.verified = False
        for object_iter in xmltree.findall('object'):
            bndbox = object_iter.find("bndbox")
            label = object_iter.find('name').text
            difficult = False
            if object_iter.find('difficult') is not None:
                difficult = bool(int(object_iter.find('difficult').text))
            self.addShape(label, bndbox, difficult)
        return True


class LoadImages:  # for inference
    def __init__(self, path, img_size=416, half=False):
        path = str(Path(path))  # os-agnostic
        files = []
        if os.path.isdir(path):
            files = sorted(glob.glob(os.path.join(path, '*.*')))
        elif os.path.isfile(path):
            files = [path]
        images = [x for x in files if os.path.splitext(x)[-1].lower() in img_formats]
        videos = [x for x in files if os.path.splitext(x)[-1].lower() in vid_formats]
        nI, nV = len(images), len(videos)
        self.img_size = img_size
        self.files = images + videos
        self.nF = nI + nV  # number of files
        self.video_flag = [False] * nI + [True] * nV
        self.mode = 'images'
        self.half = half  # half precision fp16 images
        if any(videos):
            self.new_video(videos[0])  # new video
        else:
            self.cap = None
        assert self.nF > 0, 'No images or videos found in ' + path
    def __iter__(self):
        self.count = 0
        return self
    def __next__(self):
        if self.count == self.nF:
            raise StopIteration
        path = self.files[self.count]
        if self.video_flag[self.count]:
            # Read video
            self.mode = 'video'
            ret_val, img0 = self.cap.read()
            if not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nF:  # last video
                    raise StopIteration
                else:
                    path = self.files[self.count]
                    self.new_video(path)
                    ret_val, img0 = self.cap.read()
            self.frame += 1
            print('video %g/%g (%g/%g) %s: ' % (self.count + 1, self.nF, self.frame, self.nframes, path), end='')
        else:
            # Read image
            self.count += 1
            img0 = cv2.imread(path)  # BGR
            assert img0 is not None, 'Image Not Found ' + path
            print('image %g/%g %s: ' % (self.count, self.nF, path), end='')
        # Padded resize
        img = letterbox(img0, new_shape=self.img_size)[0]
        # Normalize RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
        img = np.ascontiguousarray(img, dtype=np.float16 if self.half else np.float32)  # uint8 to fp16/fp32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        # cv2.imwrite(path + '.letterbox.jpg', 255 * img.transpose((1, 2, 0))[:, :, ::-1])  # save letterbox image
        return path, img, img0, self.cap
    def new_video(self, path):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.nframes = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
    def __len__(self):
        return self.nF  # number of files

class MediaFiles:  
    def __init__(self, path):
        path = str(Path(path)) 
        files = []
        if os.path.isdir(path):
            files = sorted(glob.glob(os.path.join(path, '*.*')))
        elif os.path.isfile(path):
            files = [path]
        images = [x for x in files if os.path.splitext(x)[-1].lower() in img_formats]
        videos = [x for x in files if os.path.splitext(x)[-1].lower() in vid_formats]
        nI, nV = len(images), len(videos)
        self.files = images + videos
        self.nF = nI + nV 
        self.video_flag = [False] * nI + [True] * nV
        self.mode = 'images'
        if any(videos):
            self.new_video(videos[0]) 
        else:
            self.cap = None
        assert self.nF > 0, 'No images or videos found in ' + path
    def __iter__(self):
        self.count = 0
        return self
    def __next__(self):
        if self.count == self.nF:
            raise StopIteration
        path = self.files[self.count]
        if self.video_flag[self.count]:
            self.mode = 'video'
            ret_val, img = self.cap.read()
            if not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nF:  
                    raise StopIteration
                else:
                    path = self.files[self.count]
                    self.new_video(path)
                    ret_val, img = self.cap.read()
            self.frame += 1
            print('video %g/%g (%g/%g) %s: ' % (self.count + 1,\
                                                self.nF, self.frame,\
                                                self.nframes, path) , end='\n')
        else:
            self.count += 1
            img = cv2.imread(path) 
            assert img is not None, 'Image Not Found ' + path
            print('image %g/%g %s: ' % (self.count, self.nF, path), end='')
        return [path], [img], self.cap
    def new_video(self, path):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.nframes = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
    def __len__(self):
        return self.nF 

class LiveFeed: 
    def __init__(self, sources='streams.txt'):
        self.mode = 'images'
        if os.path.isfile(sources):
            with open(sources, 'r') as f:
                sources = [x.strip() for x in f.read().splitlines() if len(x.strip())]
        else:
            sources = [sources]
        n = len(sources)
        self.imgs = [None] * n
        self.sources = sources
        for i, s in enumerate(sources):
            print('%g/%g: %s... ' % (i + 1, n, s), end='')
            try:
                cap = cv2.VideoCapture(int(s))
            except Exception as e:
                cap = cv2.VideoCapture(s, cv2.CAP_GSTREAMER)
            assert cap.isOpened(), 'Failed to open %s' % s
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) % 100
            _, self.imgs[i] = cap.read()
            thread = Thread(target=self.update, args=([i, cap]), daemon=True)
            print(' success (%gx%g at %.2f FPS).' % (w, h, fps))
            thread.start()
        print() 
    def update(self, index, cap):
        n = 0
        while cap.isOpened():
            n += 1
            # ret, self.imgs[index] = cap.read()
            cap.grab()
            if n == 4:  
                ret, self.imgs[index] = cap.retrieve()
                n = 0
            time.sleep(0.01)
    def __iter__(self):
        self.count = -1
        return self
    def __next__(self):
        self.count += 1
        img = self.imgs.copy()
        if cv2.waitKey(1) == ord('q'):  
            cv2.destroyAllWindows()
            raise StopIteration
        return self.sources, img, None
    def __len__(self):
        return 0

class LoadWebcam:  # for inference
    def __init__(self, pipe=0, img_size=416, half=False):
        self.img_size = img_size
        self.half = half  # half precision fp16 images
        if pipe == '0':
            pipe = 0  # local camera
        # pipe = 'rtsp://192.168.1.64/1'  # IP camera
        # pipe = 'rtsp://username:password@192.168.1.64/1'  # IP camera with login
        # pipe = 'rtsp://170.93.143.139/rtplive/470011e600ef003a004ee33696235daa'  # IP traffic camera
        # pipe = 'http://wmccpinetop.axiscam.net/mjpg/video.mjpg'  # IP golf camera
        # https://answers.opencv.org/question/215996/changing-gstreamer-pipeline-to-opencv-in-pythonsolved/
        # pipe = '"rtspsrc location="rtsp://username:password@192.168.1.64/1" latency=10 ! appsink'  # GStreamer
        # https://answers.opencv.org/question/200787/video-acceleration-gstremer-pipeline-in-videocapture/
        # https://stackoverflow.com/questions/54095699/install-gstreamer-support-for-opencv-python-package  # install help
        # pipe = "rtspsrc location=rtsp://root:root@192.168.0.91:554/axis-media/media.amp?videocodec=h264&resolution=3840x2160 protocols=GST_RTSP_LOWER_TRANS_TCP ! rtph264depay ! queue ! vaapih264dec ! videoconvert ! appsink"  # GStreamer
        self.pipe = pipe
        self.cap = cv2.VideoCapture(pipe)  # video capture object
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # set buffer size
    def __iter__(self):
        self.count = -1
        return self
    def __next__(self):
        self.count += 1
        if cv2.waitKey(1) == ord('q'):  # q to quit
            self.cap.release()
            cv2.destroyAllWindows()
            raise StopIteration
        # Read frame
        if self.pipe == 0:  # local camera
            ret_val, img0 = self.cap.read()
            img0 = cv2.flip(img0, 1)  # flip left-right
        else:  # IP camera
            n = 0
            while True:
                n += 1
                self.cap.grab()
                if n % 30 == 0:  # skip frames
                    ret_val, img0 = self.cap.retrieve()
                    if ret_val:
                        break
        # Print
        assert ret_val, 'Camera Error %s' % self.pipe
        img_path = 'webcam.jpg'
        print('webcam %g: ' % self.count, end='')
        # Padded resize
        img = letterbox(img0, new_shape=self.img_size)[0]
        # Normalize RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
        img = np.ascontiguousarray(img, dtype=np.float16 if self.half else np.float32)  # uint8 to fp16/fp32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        return img_path, img, img0, None
    def __len__(self):
        return 0

class LoadStreams:  # multiple IP or RTSP cameras
    def __init__(self, sources='streams.txt', img_size=416, half=False):
        self.mode = 'images'
        self.img_size = img_size
        self.half = half  # half precision fp16 images
        if os.path.isfile(sources):
            with open(sources, 'r') as f:
                sources = [x.strip() for x in f.read().splitlines() if len(x.strip())]
        else:
            sources = [sources]
        n = len(sources)
        self.imgs = [None] * n
        self.sources = sources
        for i, s in enumerate(sources):
            # Start the thread to read frames from the video stream
            print('%g/%g: %s... ' % (i + 1, n, s), end='')
            if s == '0':
                cap = cv2.VideoCapture(0)
            else:
                cap = cv2.VideoCapture(s, cv2.CAP_GSTREAMER)
            assert cap.isOpened(), 'Failed to open %s' % s
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) % 100
            _, self.imgs[i] = cap.read()  # guarantee first frame
            thread = Thread(target=self.update, args=([i, cap]), daemon=True)
            print(' success (%gx%g at %.2f FPS).' % (w, h, fps))
            thread.start()
        print('')  # newline
    def update(self, index, cap):
        # Read next stream frame in a daemon thread
        n = 0
        while cap.isOpened():
            n += 1
            # _, self.imgs[index] = cap.read()
            cap.grab()
            if n == 4:  # read every 4th frame
                _, self.imgs[index] = cap.retrieve()
                n = 0
            time.sleep(0.01)  # wait time
    def __iter__(self):
        self.count = -1
        return self
    def __next__(self):
        self.count += 1
        img0 = self.imgs.copy()
        if cv2.waitKey(1) == ord('q'):  # q to quit
            cv2.destroyAllWindows()
            raise StopIteration
        # Letterbox
        img = [letterbox(x, new_shape=self.img_size, interp=cv2.INTER_LINEAR)[0] for x in img0]
        # Stack
        img = np.stack(img, 0)
        # Normalize RGB
        img = img[:, :, :, ::-1].transpose(0, 3, 1, 2)  # BGR to RGB
        img = np.ascontiguousarray(img, dtype=np.float16 if self.half else np.float32)  # uint8 to fp16/fp32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        return self.sources, img, img0, None
    def __len__(self):
        return 0  # 1E12 frames = 32 streams at 30 FPS for 30 years

class LoadImagesAndLabels(Dataset):  # for training/testing
    def __init__(self, path, img_size=416, batch_size=16, augment=False, hyp=None, rect=True, image_weights=False,
                 cache_labels=False, cache_images=False):
        path = str(Path(path))  # os-agnostic
        with open(path, 'r') as f:
            self.img_files = [x.replace('/', os.sep) for x in f.read().splitlines()  # os-agnostic
                              if os.path.splitext(x)[-1].lower() in img_formats]
        n = len(self.img_files)
        bi = np.floor(np.arange(n) / batch_size).astype(np.int)  # batch index
        nb = bi[-1] + 1  # number of batches
        assert n > 0, 'No images found in %s' % path
        self.n = n
        self.batch = bi  # batch index of image
        self.img_size = img_size
        self.augment = augment
        self.hyp = hyp
        self.image_weights = image_weights
        self.rect = False if image_weights else rect
        # Define labels
        self.label_files = [x.replace('images', 'labels').replace(os.path.splitext(x)[-1], '.txt')
                            for x in self.img_files]
        # Rectangular Training  https://github.com/ultralytics/yolov3/issues/232
        if self.rect:
            # Read image shapes
            sp = 'data' + os.sep + path.replace('.txt', '.shapes').split(os.sep)[-1]  # shapefile path
            try:
                with open(sp, 'r') as f:  # read existing shapefile
                    s = [x.split() for x in f.read().splitlines()]
                    assert len(s) == n, 'Shapefile out of sync'
            except:
                s = [exif_size(Image.open(f)) for f in tqdm(self.img_files, desc='Reading image shapes')]
                np.savetxt(sp, s, fmt='%g')  # overwrites existing (if any)
            # Sort by aspect ratio
            s = np.array(s, dtype=np.float64)
            ar = s[:, 1] / s[:, 0]  # aspect ratio
            i = ar.argsort()
            self.img_files = [self.img_files[i] for i in i]
            self.label_files = [self.label_files[i] for i in i]
            self.shapes = s[i]
            ar = ar[i]
            # Set training image shapes
            shapes = [[1, 1]] * nb
            for i in range(nb):
                ari = ar[bi == i]
                mini, maxi = ari.min(), ari.max()
                if maxi < 1:
                    shapes[i] = [maxi, 1]
                elif mini > 1:
                    shapes[i] = [1, 1 / mini]
            self.batch_shapes = np.ceil(np.array(shapes) * img_size / 32.).astype(np.int) * 32
        # Preload labels (required for weighted CE training)
        self.imgs = [None] * n
        self.labels = [None] * n
        if cache_labels or image_weights:  # cache labels for faster training
            self.labels = [np.zeros((0, 5))] * n
            extract_bounding_boxes = False
            create_datasubset = False
            pbar = tqdm(self.label_files, desc='Reading labels')
            nm, nf, ne, ns = 0, 0, 0, 0  # number missing, number found, number empty, number datasubset
            for i, file in enumerate(pbar):
                try:
                    with open(file, 'r') as f:
                        l = np.array([x.split() for x in f.read().splitlines()], dtype=np.float32)
                except:
                    nm += 1  # print('missing labels for image %s' % self.img_files[i])  # file missing
                    continue
                if l.shape[0]:
                    assert l.shape[1] == 5, '> 5 label columns: %s' % file
                    assert (l >= 0).all(), 'negative labels: %s' % file
                    assert (l[:, 1:] <= 1).all(), 'non-normalized or out of bounds coordinate labels: %s' % file
                    self.labels[i] = l
                    nf += 1  # file found
                    # Create subdataset (a smaller dataset)
                    if create_datasubset and ns < 1E4:
                        if ns == 0:
                            create_folder(path='./datasubset')
                            os.makedirs('./datasubset/images')
                        exclude_classes = 43
                        if exclude_classes not in l[:, 0]:
                            ns += 1
                            # shutil.copy(src=self.img_files[i], dst='./datasubset/images/')  # copy image
                            with open('./datasubset/images.txt', 'a') as f:
                                f.write(self.img_files[i] + '\n')
                    # Extract object detection boxes for a second stage classifier
                    if extract_bounding_boxes:
                        p = Path(self.img_files[i])
                        img = cv2.imread(str(p))
                        h, w = img.shape[:2]
                        for j, x in enumerate(l):
                            f = '%s%sclassifier%s%g_%g_%s' % (p.parent.parent, os.sep, os.sep, x[0], j, p.name)
                            if not os.path.exists(Path(f).parent):
                                os.makedirs(Path(f).parent)  # make new output folder
                            b = x[1:] * np.array([w, h, w, h])  # box
                            b[2:] = b[2:].max()  # rectangle to square
                            b[2:] = b[2:] * 1.3 + 30  # pad
                            b = xywh2xyxy(b.reshape(-1, 4)).ravel().astype(np.int)
                            b[[0, 2]] = np.clip(b[[0, 2]], 0, w)  # clip boxes outside of image
                            b[[1, 3]] = np.clip(b[[1, 3]], 0, h)
                            assert cv2.imwrite(f, img[b[1]:b[3], b[0]:b[2]]), 'Failure extracting classifier boxes'
                else:
                    ne += 1  # print('empty labels for image %s' % self.img_files[i])  # file empty
                    # os.system("rm '%s' '%s'" % (self.img_files[i], self.label_files[i]))  # remove
                pbar.desc = 'Reading labels (%g found, %g missing, %g empty for %g images)' % (nf, nm, ne, n)
            assert nf > 0, 'No labels found. Recommend correcting image and label paths.'
        # Cache images into memory for faster training (~5GB)
        if cache_images and augment:  # if training
            for i in tqdm(range(min(len(self.img_files), 10000)), desc='Reading images'):  # max 10k images
                img_path = self.img_files[i]
                img = cv2.imread(img_path)  # BGR
                assert img is not None, 'Image Not Found ' + img_path
                r = self.img_size / max(img.shape)  # size ratio
                if self.augment and r < 1:  # if training (NOT testing), downsize to inference shape
                    h, w = img.shape[:2]
                    img = cv2.resize(img, (int(w * r), int(h * r)), interpolation=cv2.INTER_LINEAR)  # or INTER_AREA
                self.imgs[i] = img
        # Detect corrupted images https://medium.com/joelthchao/programmatically-detect-corrupted-image-8c1b2006c3d3
        detect_corrupted_images = False
        if detect_corrupted_images:
            from skimage import io  # conda install -c conda-forge scikit-image
            for file in tqdm(self.img_files, desc='Detecting corrupted images'):
                try:
                    _ = io.imread(file)
                except:
                    print('Corrupted image detected: %s' % file)
    def __len__(self):
        return len(self.img_files)
    # def __iter__(self):
    #     self.count = -1
    #     print('ran dataset iter')
    #     #self.shuffled_vector = np.random.permutation(self.nF) if self.augment else np.arange(self.nF)
    #     return self
    def __getitem__(self, index):
        if self.image_weights:
            index = self.indices[index]
        img_path = self.img_files[index]
        label_path = self.label_files[index]
        hyp = self.hyp
        mosaic = True and self.augment  # load 4 images at a time into a mosaic (only during training)
        if mosaic:
            # Load mosaic
            img, labels = load_mosaic(self, index)
            h, w = img.shape[:2]
            ratio, pad = None, None
        else:
            # Load image
            img = load_image(self, index)
            # Letterbox
            h, w = img.shape[:2]
            shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size  # final letterboxed shape
            img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)
            # Load labels
            labels = []
            if os.path.isfile(label_path):
                x = self.labels[index]
                if x is None:  # labels not preloaded
                    with open(label_path, 'r') as f:
                        x = np.array([x.split() for x in f.read().splitlines()], dtype=np.float32)
                if x.size > 0:
                    # Normalized xywh to pixel xyxy format
                    labels = x.copy()
                    labels[:, 1] = ratio[0] * w * (x[:, 1] - x[:, 3] / 2) + pad[0]  # pad width
                    labels[:, 2] = ratio[1] * h * (x[:, 2] - x[:, 4] / 2) + pad[1]  # pad height
                    labels[:, 3] = ratio[0] * w * (x[:, 1] + x[:, 3] / 2) + pad[0]
                    labels[:, 4] = ratio[1] * h * (x[:, 2] + x[:, 4] / 2) + pad[1]
        if self.augment:
            # Augment imagespace
            if not mosaic:
                img, labels = random_affine(img, labels,
                                            degrees=hyp['degrees'],
                                            translate=hyp['translate'],
                                            scale=hyp['scale'],
                                            shear=hyp['shear'])
            # Augment colorspace
            augment_hsv(img, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v'])
            # Apply cutouts
            # if random.random() < 0.9:
            #     labels = cutout(img, labels)
        nL = len(labels)  # number of labels
        if nL:
            # convert xyxy to xywh
            labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])
            # Normalize coordinates 0 - 1
            labels[:, [2, 4]] /= img.shape[0]  # height
            labels[:, [1, 3]] /= img.shape[1]  # width
        if self.augment:
            # random left-right flip
            lr_flip = True
            if lr_flip and random.random() < 0.5:
                img = np.fliplr(img)
                if nL:
                    labels[:, 1] = 1 - labels[:, 1]
            # random up-down flip
            ud_flip = False
            if ud_flip and random.random() < 0.5:
                img = np.flipud(img)
                if nL:
                    labels[:, 2] = 1 - labels[:, 2]
        labels_out = torch.zeros((nL, 6))
        if nL:
            labels_out[:, 1:] = torch.from_numpy(labels)
        # Normalize
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img, dtype=np.float32)  # uint8 to float32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        return torch.from_numpy(img), labels_out, img_path, ((h, w), (ratio, pad))
    @staticmethod
    def collate_fn(batch):
        img, label, path, shapes = list(zip(*batch))  # transposed
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0), path, shapes

class TrainingDataset(Dataset):
    def __init__(self,
                 path,
                 img_size=416,
                 batch_size=16,
                 augment=False,
                 aug_hyp=None,
                 rect=True,
                 image_weights=False,
                 cache_labels=False,
                 cache_images=False):
        path = str(Path(path))
        name = os.path.basename(path)
        dataset_name = path.split(os.sep)[-2]
        self.load_paths(path)
        num_imgs = len(self.img_files)
        batch_idx = np.floor(np.arange(num_imgs) / batch_size).astype(np.int)
        num_batches = batch_idx[-1] + 1
        assert num_imgs > 0, 'No images found in %s' % path
        self.num_imgs = num_imgs
        self.batch = batch_idx
        self.img_size = img_size
        self.augment = augment
        self.aug_hyp = aug_hyp
        self.image_weights = image_weights
        self.rect = False if image_weights else rect

        if self.rect:
            #shape_path = 'data' + os.sep + path.replace('.txt', '.shapes').split(os.sep)[-1]
            shape_path = os.path.join('data', dataset_name, name + '.shapes')
            try:
                with open(shape_path, 'r') as f:
                    shapes_array = [x.split() for x in f.read().splitlines()]
                    assert len(shapes_array) == num_imgs, 'Shapefile out of sync'
            except:
                shapes_array = [exif_size(Image.open(f)) \
                                for f in tqdm(self.img_files, desc='Reading image shapes')]
                np.savetxt(shape_path, shapes_array, fmt='%g')
            shapes_array = np.array(shapes_array, dtype=np.float64)
            aspect_ratios = shapes_array[:, 1] / shapes_array[:, 0]
            sorted_ar_idx = aspect_ratios.argsort()
            self.img_files = [self.img_files[i] for i in sorted_ar_idx]
            self.label_files = [self.label_files[i] for i in sorted_ar_idx]
            self.shapes = shapes_array[sorted_ar_idx]
            aspect_ratios = aspect_ratios[sorted_ar_idx]
            training_shapes = [[1, 1]] * num_batches
            for i in range(num_batches):
                aspect_ratio = aspect_ratios[batch_idx == i]
                minimum, maximum = aspect_ratio.min(), aspect_ratio.max()
                if maximum < 1:
                    training_shapes[i] = [maximum, 1]
                elif minimum > 1:
                    training_shapes[i] = [1, 1 / minimum]
            self.batch_shapes = np.ceil(np.array(training_shapes) * img_size / 32.).astype(np.int) * 32

        # Preload labels (required for weighted CE training)
        self.imgs = [None] * num_imgs
        self.labels = [None] * num_imgs

        if cache_labels or image_weights:
            self.labels = [np.zeros((0, 5))] * num_imgs
            extract_bounding_boxes = False
            create_datasubset = False
            pbar = tqdm(self.label_files, desc='Reading labels')
            num_miss, num_found, num_empty, num_subset = 0, 0, 0, 0
            for i, file in enumerate(pbar):
                try:
                    label_shape = np.array(self.convert2yoloformat(file), dtype=np.float32)
                except:
                    num_miss += 1
                    continue
                if label_shape.shape[0]:
                    assert label_shape.shape[1] == 5, '> 5 label columns: %s' % file
                    assert (label_shape >= 0).all(), 'negative labels: %s' % file
                    assert (label_shape[:, 1:] <= 1).all(), 'non-normalized or out of bounds coordinate labels: %s' % file
                    self.labels[i] = label_shape
                    num_found += 1
                    if create_datasubset and num_subset < 1E4:
                        if num_subset == 0:
                            create_folder(path='./datasubset')
                            os.makedirs('./datasubset/images')
                        exclude_classes = 43
                        if exclude_classes not in label_shape[:, 0]:
                            num_subset += 1
                            # shutil.copy(src=self.img_files[i], dst='./datasubset/images/')  # copy image
                            with open('./datasubset/images.txt', 'a') as f:
                                f.write(self.img_files[i] + '\n')
                    if extract_bounding_boxes:
                        p = Path(self.img_files[i])
                        img = cv2.imread(str(p))
                        h, w = img.shape[:2]
                        for j, x in enumerate(label_shape):
                            f = '%s%sclassifier%s%g_%g_%s' % (p.parent.parent, os.sep, os.sep, x[0], j, p.name)
                            if not os.path.exists(Path(f).parent):
                                os.makedirs(Path(f).parent)
                            box = x[1:] * np.array([w, h, w, h]) 
                            box[2:] = box[2:].max()  
                            box[2:] = box[2:] * 1.3 + 30
                            box = xywh2xyxy(box.reshape(-1, 4)).ravel().astype(np.int)
                            box[[0, 2]] = np.clip(box[[0, 2]], 0, w)
                            box[[1, 3]] = np.clip(b[[1, 3]], 0, h)
                            assert cv2.imwrite(f, img[box[1]:box[3], box[0]:box[2]]), 'Failure extracting classifier boxes'
                else:
                    num_empty += 1
                    print('empty labels for image %s' % self.img_files[i])
                    # os.system("rm '%s' '%s'" % (self.img_files[i], self.label_files[i]))
                pbar.desc = 'Reading labels (%g found, %g missing, %g empty for %g images)' % (num_found, num_miss, num_empty, num_imgs)
            assert num_found > 0, 'No labels found. Recommend correcting image and label paths.'

        if cache_images and augment:
            for i in tqdm(range(min(len(self.img_files), 10000)), desc='Reading images'):
                img_path = self.img_files[i]
                img = cv2.imread(img_path)
                assert img is not None, 'Image Not Found ' + img_path
                ratio = self.img_size / max(img.shape)
                if self.augment and ratio < 1:
                    h, w = img.shape[:2]
                    img = cv2.resize(img, (int(w * ratio),
                                           int(h * ratio)),
                                     interpolation=cv2.INTER_LINEAR)  #INTER_AREA
                self.imgs[i] = img
        detect_corrupted_images = False
        if detect_corrupted_images:
            from skimage import io
            for file in tqdm(self.img_files, desc='Detecting corrupted images'):
                try:
                    _ = io.imread(file)
                except:
                    print('Corrupted image detected: %s' % file)
    def __len__(self):
        return len(self.img_files)
    # def __iter__(self):
    #     self.count = -1
    #     print('ran dataset iter')
    #     #self.shuffled_vector = np.random.permutation(self.nF) if self.augment else np.arange(self.nF)
    #     return self
    def __getitem__(self, index):
        if self.image_weights:
            index = self.indices[index]
        img_path = self.img_files[index]
        label_path = self.label_files[index]
        aug_hyp = self.aug_hyp
        mosaic = True and self.augment
        if mosaic:
            img, labels = self.load_mosaic(index)
            h, w = img.shape[:2]
            ratio, pad = None, None
        else:
            img = self.load_image(index)
            h, w = img.shape[:2]
            shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size
            img, ratio, pad = self.letterbox(img, shape, auto=False, scaleup=self.augment)
            labels = []
            if os.path.isfile(label_path):
                label = self.labels[index]
                if label is None:
                    label = np.array(self.convert2yoloformat(label_path), dtype=np.float32)
                if label.size > 0:
                    labels = label.copy()
                    labels[:, 1] = ratio[0] * w * (label[:, 1] - label[:, 3] / 2) + pad[0] 
                    labels[:, 2] = ratio[1] * h * (label[:, 2] - label[:, 4] / 2) + pad[1] 
                    labels[:, 3] = ratio[0] * w * (label[:, 1] + label[:, 3] / 2) + pad[0]
                    labels[:, 4] = ratio[1] * h * (label[:, 2] + label[:, 4] / 2) + pad[1]
        if self.augment:
            if not mosaic:
                img, labels = self.random_affine(img, labels,
                                                 degrees=aug_hyp['degrees'],
                                                 translate=aug_hyp['translate'],
                                                 scale=aug_hyp['scale'],
                                                 shear=aug_hyp['shear'])
            augment_hsv(img, hgain=aug_hyp['hsv_h'], sgain=aug_hyp['hsv_s'], vgain=aug_hyp['hsv_v'])
            # Apply cutouts
            # if random.random() < 0.9:
            #     labels = self.cutout(img, labels)
        num_labels = len(labels)
        if num_labels:
            labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])
            labels[:, [2, 4]] /= img.shape[0] 
            labels[:, [1, 3]] /= img.shape[1]
        if self.augment:
            lr_flip = True
            if lr_flip and random.random() < 0.5:
                img = np.fliplr(img)
                if num_labels:
                    labels[:, 1] = 1 - labels[:, 1]
            ud_flip = False
            if ud_flip and random.random() < 0.5:
                img = np.flipud(img)
                if num_labels:
                    labels[:, 2] = 1 - labels[:, 2]
        labels_torch = torch.zeros((num_labels, 6))
        if num_labels:
            labels_torch[:, 1:] = torch.from_numpy(labels)
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img /= 255.0
        return torch.from_numpy(img), labels_torch, img_path, ((h, w), (ratio, pad))

    def load_paths(self, path):
        name = os.path.basename(path)
        dirpath = os.path.dirname(path)
        cfg_data = parse_data_cfg(dirpath + '.data')
        #rename_rand_pair(dirpath, XML_EXT, img_formats).main()
        #with codecs.open(dirpath + '.names', 'r', 'utf8') as f:
        with codecs.open(cfg_data["names"], 'r', 'utf8') as f:
            self.class_names = [line.strip() for line in f]
        self.img_files = [os.path.abspath(os.path.join(path,fi)) for fi in os.listdir(path) \
                        if os.path.splitext(fi)[-1] != XML_EXT]
        self.label_files = [p.replace(os.path.splitext(p)[-1], XML_EXT) for p in self.img_files]
        
#        try:
#            with codecs.open(os.path.join(dirpath, name + TXT_EXT)) as f:
#                self.img_files = [line.split() for line in f.read().splitlines()]
#                assert len(self.img_files) == num_imgs, 'Out of sync'
#        except:
#            self.img_files = [os.path.abspath(os.path.join(path,fi)) for fi in os.listdir(path) \
#                            if os.path.splitext(fi)[-1] != XML_EXT]
#            self.label_files = [p.replace(os.path.splitext(p)[-1], XML_EXT) for p in self.img_files]
#            with open(os.path.join(dirpath, name + TXT_EXT), 'w') as f:
#                for p in self.img_files:
#                    f.write(p + "\n")

    def convert2yoloformat(self, file_path):
        def pt2bndbox(points):
                xmin = float('inf')
                ymin = float('inf')
                xmax = float('-inf')
                ymax = float('-inf')
                for p in points:
                    x = p[0]
                    y = p[1]
                    xmin = min(x, xmin)
                    ymin = min(y, ymin)
                    xmax = max(x, xmax)
                    ymax = max(y, ymax)
                if xmin < 1:
                    xmin = 1
                if ymin < 1:
                    ymin = 1
                return (int(xmin), int(ymin), int(xmax), int(ymax))
        vocreader = PascalVocReader(file_path)
        shapes = vocreader.getShapes()
        imageShape = vocreader.getImageData()
        yoloformat_array = []
        for shape in shapes:
            label = shape[0]
            points = shape[1]
            difficult = int(shape[-1])
            bndbox = pt2bndbox(points)
            xcenter = float((bndbox[0] + bndbox[2])) / 2 / imageShape[1]
            ycenter = float((bndbox[1] + bndbox[3])) / 2 / imageShape[0]
            w = float((bndbox[2] - bndbox[0])) / imageShape[1]
            h = float((bndbox[3] - bndbox[1])) / imageShape[0]
            class_idx = self.class_names.index(label)
            yoloformat_array.append([class_idx, xcenter, ycenter, w, h])
        return yoloformat_array

    #def load_mosaic(self, index):
    #    out_labels = []
    #    img_size = self.img_size
    #    mosaic_xcen, mosaic_ycen = [int(random.uniform(img_size * 0.5, img_size * 1.5)) for _ in range(2)]
    #    out_img = np.zeros((img_size * 2, img_size * 2, 3), dtype=np.uint8) + 128  # base image with 4 tiles
    #    indices = [index] + [random.randint(0, len(self.labels) - 1) for _ in range(3)]  
    #    for i, index in enumerate(indices):
    #        img = self.load_image(index)
    #        height, width = img.shape[:2]
    #        if i == 0:  # top left
    #            xmin_large, ymin_large, xmax_large, ymax_large = \
    #                max(mosaic_xcen - width, 0), \
    #                max(mosaic_ycen - height, 0), \
    #                mosaic_xcen, \
    #                mosaic_ycen  
    #            xmin_small, ymin_small, xmax_small, ymax_small = \
    #                width - (xmax_large - xmin_large), \
    #                height - (ymax_large - ymin_large), \
    #                width, \
    #                height  
    #        elif i == 1:  # top right
    #            xmin_large, ymin_large, xmax_large, ymax_large = \
    #                mosaic_xcen, \
    #                max(mosaic_ycen - height, 0), \
    #                min(mosaic_xcen + width, img_size * 2), \
    #                mosaic_ycen
    #        #    x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
    #            xmin_small, ymin_small, xmax_small, ymax_small = \
    #                0, \
    #                height - (ymax_large - ymin_large), \
    #                min(width, xmax_large - xmin_large), \
    #                height
    #        elif i == 2:  # bottom left
    #            xmin_large, ymin_large, xmax_large, ymax_large = \
    #                max(mosaic_xcen - width, 0), \
    #                mosaic_ycen, \
    #                mosaic_xcen, \
    #                min(img_size * 2, mosaic_ycen + height)
    #            xmin_small, ymin_small, xmax_small, ymax_small = \
    #                width - (xmax_large - xmin_large), \
    #                0, \
    #                max(mosaic_xcen, width), \
    #                min(ymax_large - ymin_large, height)
    #        elif i == 3:  # bottom right
    #            xmin_large, ymin_large, xmax_large, ymax_large = \
    #                mosaic_xcen, \
    #                mosaic_ycen, \
    #                min(mosaic_xcen + width, img_size * 2), \
    #                min(img_size * 2, mosaic_ycen + height)
    #            xmin_small, ymin_small, xmax_small, ymax_small = \
    #                0, \
    #                0, \
    #                min(width, xmax_large - xmin_large), \
    #                min(ymax_large - ymin_large, height)
    #        out_img[ymin_large:ymax_large, xmin_large:xmax_large] = img[ymin_small:ymax_small, xmin_small:xmax_small]  
    #        pad_width = xmin_large - xmin_small
    #        pad_height = ymin_large - ymin_small
    #        label_path = self.label_files[index]
    #        if os.path.isfile(label_path):
    #            label = self.labels[index]
    #            label = np.array(self.convert2yoloformat(label), dtype=np.float32)
    #            if label.size > 0:
    #                labels = label.copy()
    #                labels[:, 1] = width * (label[:, 1] - label[:, 3] / 2) + pad_width
    #                labels[:, 2] = height * (label[:, 2] - label[:, 4] / 2) + pad_height
    #                labels[:, 3] = width * (label[:, 1] + label[:, 3] / 2) + pad_width
    #                labels[:, 4] = height * (label[:, 2] + label[:, 4] / 2) + pad_height
    #            else:
    #                labels = np.zeros((0, 5), dtype=np.float32)
    #            out_labels.append(labels)
    #    if len(out_labels):
    #        out_labels = np.concatenate(out_labels, 0)
    #        np.clip(out_labels[:, 1:], 0, 2 * img_size, out=out_labels[:, 1:])
    #    out_img, out_labels = self.random_affine(out_img,
    #                                             out_labels,
    #                                             degrees=self.aug_hyp['degrees'],
    #                                             translate=self.aug_hyp['translate'],
    #                                             scale=self.aug_hyp['scale'],
    #                                             shear=self.aug_hyp['shear'],
    #                                             border=-img_size//2)  # border to remove
    #    return out_img, out_labels

    def load_mosaic(self, index):
        # loads images in a mosaic
        labels4 = []
        s = self.img_size
        xc, yc = [int(random.uniform(s * 0.5, s * 1.5)) for _ in range(2)]  # mosaic center x, y
        img4 = np.zeros((s * 2, s * 2, 3), dtype=np.uint8) + 128  # base image with 4 tiles
        # 3 additional image indices
        indices = [index] + [random.randint(0, len(self.labels) - 1) for _ in range(3)]  
        for i, index in enumerate(indices):
            # Load image
            img = load_image(self, index)
            h, w, _ = img.shape
            # place img in img4
            if i == 0:  # top left
                # xmin, ymin, xmax, ymax (large image)
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  
                # xmin, ymin, xmax, ymax (small image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            padw = x1a - x1b
            padh = y1a - y1b
            # Load labels
            label_path = self.label_files[index]
            if os.path.isfile(label_path):
                x = self.labels[index]
                if x is None:  # labels not preloaded
                    with open(label_path, 'r') as f:
                        x = np.array([x.split() for x in f.read().splitlines()], dtype=np.float32)
                if x.size > 0:
                    # Normalized xywh to pixel xyxy format
                    labels = x.copy()
                    labels[:, 1] = w * (x[:, 1] - x[:, 3] / 2) + padw
                    labels[:, 2] = h * (x[:, 2] - x[:, 4] / 2) + padh
                    labels[:, 3] = w * (x[:, 1] + x[:, 3] / 2) + padw
                    labels[:, 4] = h * (x[:, 2] + x[:, 4] / 2) + padh
                else:
                    labels = np.zeros((0, 5), dtype=np.float32)
                labels4.append(labels)
        # Concat/clip labels
        if len(labels4):
            labels4 = np.concatenate(labels4, 0)
            np.clip(labels4[:, 1:], 0, 2 * s, out=labels4[:, 1:])
        # Augment
        img4, labels4 = random_affine(img4, labels4,
                                    degrees=self.aug_hyp['degrees'],
                                    translate=self.aug_hyp['translate'],
                                    scale=self.aug_hyp['scale'],
                                    shear=self.aug_hyp['shear'],
                                    border=-s // 2)  # border to remove
        return img4, labels4

    def load_image(self, index):
        # loads 1 image from dataset
        img = self.imgs[index]
        if img is None:
            img_path = self.img_files[index]
            img = cv2.imread(img_path)  # BGR
            assert img is not None, 'Image Not Found ' + img_path
            r = self.img_size / max(img.shape)  # size ratio
            if self.augment:  # if training (NOT testing), downsize to inference shape
                h, w = img.shape[:2]
                # _LINEAR fastest
                img = cv2.resize(img, (int(w * r), int(h * r)), interpolation=cv2.INTER_LINEAR)  
        return img

    def letterbox(self, img, new_shape=(416, 416), color=(128, 128, 128),
                auto=True, scaleFill=False, scaleup=True, interp=cv2.INTER_AREA):
        # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)
        # Scale ratio (new / old)
        r = max(new_shape) / max(shape)
        if not scaleup:  # only scale down, do not scale up (for better test mAP)
            r = min(r, 1.0)
        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = new_shape
            ratio = new_shape[0] / shape[1], new_shape[1] / shape[0]  # width, height ratios
        dw /= 2  # divide padding into 2 sides
        dh /= 2
        if shape[::-1] != new_unpad:  # resize
            # INTER_AREA is better, INTER_LINEAR is faster
            img = cv2.resize(img, new_unpad, interpolation=interp)  
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left,
                                right, cv2.BORDER_CONSTANT, value=color)  # add border
        return img, ratio, (dw, dh)

    def cutout(self, image, labels):
        def bbox_ioa(box1, box2, x1y1x2y2=True):
            box2 = box2.transpose()
            inter_area = \
                ( np.minimum(box1[2], box2[2]) - np.maximum(box1[0], box2[0]) ).clip(0) * \
                ( np.minimum(box1[3], box2[3]) - np.maximum(box1[1], box2[1]) ).clip(0)
            box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1]) + 1e-16
            box2_iou = inter_area / box2_area
            return box2_iou
        h, w = image.shape[:2]
        # create random masks
        scales = [0.5] * 1  \
            # + [0.25] * 4 + [0.125] * 16 + [0.0625] * 64 + [0.03125] * 256  # image size fraction
        for s in scales:
            mask_h = random.randint(1, int(h * s))
            mask_w = random.randint(1, int(w * s))
            xmin = max(0, random.randint(0, w) - mask_w // 2)
            ymin = max(0, random.randint(0, h) - mask_h // 2)
            xmax = min(w, xmin + mask_w)
            ymax = min(h, ymin + mask_h)
            mask_color = [random.randint(0, 255) for _ in range(3)]
            image[ymin:ymax, xmin:xmax] = mask_color
            # return unobscured labels
            if len(labels) and s > 0.03:
                box = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
                ioa = bbox_ioa(box, labels[:, 1:5])
                labels = labels[ioa < 0.90]
        return labels

    def random_affine(self, img, targets=(), degrees=10, translate=.1, scale=.1, shear=10, border=0):
        # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
        # https://medium.com/uruvideo/dataset-augmentation-with-random-homographies-a8f4b44830d4
        if targets is None:  # targets = [cls, xyxy]
            targets = []
        height = img.shape[0] + border * 2
        width = img.shape[1] + border * 2
        # Rotation and Scale
        R = np.eye(3)
        a = random.uniform(-degrees, degrees)
        # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
        s = random.uniform(1 - scale, 1 + scale)
        R[:2] = cv2.getRotationMatrix2D(angle=a, center=(img.shape[1] / 2, img.shape[0] / 2), scale=s)
        # Translation
        T = np.eye(3)
        # x translation (pixels)
        T[0, 2] = random.uniform(-translate, translate) * img.shape[0] + border  
        # y translation (pixels)
        T[1, 2] = random.uniform(-translate, translate) * img.shape[1] + border  
        # Shear
        S = np.eye(3)
        S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
        S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)
        # Combined rotation matrix
        M = S @ T @ R  # ORDER IS IMPORTANT HERE!!
        changed = (border != 0) or (M != np.eye(3)).any()
        if changed:
            img = cv2.warpAffine(img, M[:2], dsize=(width, height),
                                flags=cv2.INTER_AREA, borderValue=(128, 128, 128))
        # Transform label coordinates
        n = len(targets)
        if n:
            # warp points
            xy = np.ones((n * 4, 3))
            # x1y1, x2y2, x1y2, x2y1
            xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)
            xy = (xy @ M.T)[:, :2].reshape(n, 8)
            # create new boxes
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T
            # # apply angle-based reduction of bounding boxes
            # radians = a * math.pi / 180
            # reduction = max(abs(math.sin(radians)), abs(math.cos(radians))) ** 0.5
            # x = (xy[:, 2] + xy[:, 0]) / 2
            # y = (xy[:, 3] + xy[:, 1]) / 2
            # w = (xy[:, 2] - xy[:, 0]) * reduction
            # h = (xy[:, 3] - xy[:, 1]) * reduction
            # xy = np.concatenate((x - w / 2, y - h / 2, x + w / 2, y + h / 2)).reshape(4, n).T
            # reject warped points outside of image
            xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
            xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)
            w = xy[:, 2] - xy[:, 0]
            h = xy[:, 3] - xy[:, 1]
            area = w * h
            area0 = (targets[:, 3] - targets[:, 1]) * (targets[:, 4] - targets[:, 2])
            ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))
            i = (w > 4) & (h > 4) & (area / (area0 + 1e-16) > 0.1) & (ar < 10)
            targets = targets[i]
            targets[:, 1:5] = xy[i]
        return img, targets

    @staticmethod
    def collate_fn(batch):
        img, labels, path, shapes = list(zip(*batch))
        for i, label in enumerate(labels):
            label[:, 0] = i
        return torch.stack(img, 0), torch.cat(labels, 0), path, shapes

def load_mosaic(self, index):
    # loads images in a mosaic
    labels4 = []
    s = self.img_size
    xc, yc = [int(random.uniform(s * 0.5, s * 1.5)) for _ in range(2)]  # mosaic center x, y
    img4 = np.zeros((s * 2, s * 2, 3), dtype=np.uint8) + 128  # base image with 4 tiles
    # 3 additional image indices
    indices = [index] + [random.randint(0, len(self.labels) - 1) for _ in range(3)]  
    for i, index in enumerate(indices):
        # Load image
        img = load_image(self, index)
        h, w, _ = img.shape
        # place img in img4
        if i == 0:  # top left
            # xmin, ymin, xmax, ymax (large image)
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  
            # xmin, ymin, xmax, ymax (small image)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  
        elif i == 1:  # top right
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom left
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
        elif i == 3:  # bottom right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
        img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        padw = x1a - x1b
        padh = y1a - y1b
        # Load labels
        label_path = self.label_files[index]
        if os.path.isfile(label_path):
            x = self.labels[index]
            if x is None:  # labels not preloaded
                with open(label_path, 'r') as f:
                    x = np.array([x.split() for x in f.read().splitlines()], dtype=np.float32)
            if x.size > 0:
                # Normalized xywh to pixel xyxy format
                labels = x.copy()
                labels[:, 1] = w * (x[:, 1] - x[:, 3] / 2) + padw
                labels[:, 2] = h * (x[:, 2] - x[:, 4] / 2) + padh
                labels[:, 3] = w * (x[:, 1] + x[:, 3] / 2) + padw
                labels[:, 4] = h * (x[:, 2] + x[:, 4] / 2) + padh
            else:
                labels = np.zeros((0, 5), dtype=np.float32)
            labels4.append(labels)
    # Concat/clip labels
    if len(labels4):
        labels4 = np.concatenate(labels4, 0)
        np.clip(labels4[:, 1:], 0, 2 * s, out=labels4[:, 1:])
    # Augment
    img4, labels4 = random_affine(img4, labels4,
                                degrees=self.hyp['degrees'],
                                translate=self.hyp['translate'],
                                scale=self.hyp['scale'],
                                shear=self.hyp['shear'],
                                border=-s // 2)  # border to remove
    return img4, labels4

def load_image(self, index):
    # loads 1 image from dataset
    img = self.imgs[index]
    if img is None:
        img_path = self.img_files[index]
        img = cv2.imread(img_path)  # BGR
        assert img is not None, 'Image Not Found ' + img_path
        r = self.img_size / max(img.shape)  # size ratio
        if self.augment:  # if training (NOT testing), downsize to inference shape
            h, w = img.shape[:2]
            # _LINEAR fastest
            img = cv2.resize(img, (int(w * r), int(h * r)), interpolation=cv2.INTER_LINEAR)  
    return img

def augment_hsv(img, hgain=0.5, sgain=0.5, vgain=0.5):
    # random gains
    x = (np.random.uniform(-1, 1, 3) * np.array([hgain, sgain, vgain]) + 1).astype(np.float32)  
    img_hsv = (cv2.cvtColor(img, cv2.COLOR_BGR2HSV) * \
               x.reshape((1, 1, 3))).clip(None, 255).astype(np.uint8)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed

def letterbox(img, new_shape=(416, 416), color=(128, 128, 128),
              auto=True, scaleFill=False, scaleup=True, interp=cv2.INTER_AREA):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    # Scale ratio (new / old)
    r = max(new_shape) / max(shape)
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)
    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = new_shape
        ratio = new_shape[0] / shape[1], new_shape[1] / shape[0]  # width, height ratios
    dw /= 2  # divide padding into 2 sides
    dh /= 2
    if shape[::-1] != new_unpad:  # resize
        # INTER_AREA is better, INTER_LINEAR is faster
        img = cv2.resize(img, new_unpad, interpolation=interp)  
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left,
                             right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

def random_affine(img, targets=(), degrees=10, translate=.1, scale=.1, shear=10, border=0):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # https://medium.com/uruvideo/dataset-augmentation-with-random-homographies-a8f4b44830d4
    if targets is None:  # targets = [cls, xyxy]
        targets = []
    height = img.shape[0] + border * 2
    width = img.shape[1] + border * 2
    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(img.shape[1] / 2, img.shape[0] / 2), scale=s)
    # Translation
    T = np.eye(3)
    # x translation (pixels)
    T[0, 2] = random.uniform(-translate, translate) * img.shape[0] + border  
    # y translation (pixels)
    T[1, 2] = random.uniform(-translate, translate) * img.shape[1] + border  
    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)
    # Combined rotation matrix
    M = S @ T @ R  # ORDER IS IMPORTANT HERE!!
    changed = (border != 0) or (M != np.eye(3)).any()
    if changed:
        img = cv2.warpAffine(img, M[:2], dsize=(width, height),
                             flags=cv2.INTER_AREA, borderValue=(128, 128, 128))
    # Transform label coordinates
    n = len(targets)
    if n:
        # warp points
        xy = np.ones((n * 4, 3))
        # x1y1, x2y2, x1y2, x2y1
        xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)
        xy = (xy @ M.T)[:, :2].reshape(n, 8)
        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T
        # # apply angle-based reduction of bounding boxes
        # radians = a * math.pi / 180
        # reduction = max(abs(math.sin(radians)), abs(math.cos(radians))) ** 0.5
        # x = (xy[:, 2] + xy[:, 0]) / 2
        # y = (xy[:, 3] + xy[:, 1]) / 2
        # w = (xy[:, 2] - xy[:, 0]) * reduction
        # h = (xy[:, 3] - xy[:, 1]) * reduction
        # xy = np.concatenate((x - w / 2, y - h / 2, x + w / 2, y + h / 2)).reshape(4, n).T
        # reject warped points outside of image
        xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
        xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)
        w = xy[:, 2] - xy[:, 0]
        h = xy[:, 3] - xy[:, 1]
        area = w * h
        area0 = (targets[:, 3] - targets[:, 1]) * (targets[:, 4] - targets[:, 2])
        ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))
        i = (w > 4) & (h > 4) & (area / (area0 + 1e-16) > 0.1) & (ar < 10)
        targets = targets[i]
        targets[:, 1:5] = xy[i]
    return img, targets

def cutout(image, labels):
    # https://arxiv.org/abs/1708.04552
    # https://github.com/hysts/pytorch_cutout/blob/master/dataloader.py
    # https://towardsdatascience.com/when-conventional-wisdom-fails-revisiting-data-augmentation-for-self-driving-cars-4831998c5509
    h, w = image.shape[:2]
    def bbox_ioa(box1, box2, x1y1x2y2=True):
        # Returns the intersection over box2 area given box1, box2.
        #box1 is 4, box2 is nx4. boxes are x1y1x2y2
        box2 = box2.transpose()
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
        # Intersection area
        inter_area = (np.minimum(b1_x2, b2_x2) - np.maximum(b1_x1, b2_x1)).clip(0) * \
                     (np.minimum(b1_y2, b2_y2) - np.maximum(b1_y1, b2_y1)).clip(0)
        # box2 area
        box2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1) + 1e-16
        # Intersection over box2 area
        return inter_area / box2_area
    # create random masks
    scales = [0.5] * 1  \
        # + [0.25] * 4 + [0.125] * 16 + [0.0625] * 64 + [0.03125] * 256  # image size fraction
    for s in scales:
        mask_h = random.randint(1, int(h * s))
        mask_w = random.randint(1, int(w * s))
        # box
        xmin = max(0, random.randint(0, w) - mask_w // 2)
        ymin = max(0, random.randint(0, h) - mask_h // 2)
        xmax = min(w, xmin + mask_w)
        ymax = min(h, ymin + mask_h)
        # apply random color mask
        mask_color = [random.randint(0, 255) for _ in range(3)]
        image[ymin:ymax, xmin:xmax] = mask_color
        # return unobscured labels
        if len(labels) and s > 0.03:
            box = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
            ioa = bbox_ioa(box, labels[:, 1:5])  # intersection over area
            labels = labels[ioa < 0.90]  # remove >90% obscured labels
    return labels

# from utils.datasets import *; reduce_img_size()
def reduce_img_size(path='../data/sm4/images', img_size=1024):  
    # creates a new ./images_reduced folder with reduced size images of maximum size img_size
    path_new = path + '_reduced'  # reduced images path
    create_folder(path_new)
    for f in tqdm(glob.glob('%s/*.*' % path)):
        try:
            img = cv2.imread(f)
            h, w = img.shape[:2]
            r = img_size / max(h, w)  # size ratio
            if r < 1.0:
                # _LINEAR fastest
                img = cv2.resize(img, (int(w * r), int(h * r)), interpolation=cv2.INTER_AREA)  
            fnew = f.replace(path, path_new)  # .replace(Path(f).suffix, '.jpg')
            cv2.imwrite(fnew, img)
        except:
            print('WARNING: image failure %s' % f)

def convert_images2bmp():
    # cv2.imread() jpg at 230 img/s, *.bmp at 400 img/s
    for path in ['../coco/images/val2014/', '../coco/images/train2014/']:
        folder = os.sep + Path(path).name
        output = path.replace(folder, folder + 'bmp')
        create_folder(output)
        for f in tqdm(glob.glob('%s*.jpg' % path)):
            save_name = f.replace('.jpg', '.bmp').replace(folder, folder + 'bmp')
            cv2.imwrite(save_name, cv2.imread(f))
    for label_path in ['../coco/trainvalno5k.txt', '../coco/5k.txt']:
        with open(label_path, 'r') as file:
            lines = file.read()
        lines = lines.replace('2014/', '2014bmp/').replace('.jpg', '.bmp').replace(
            '/Users/glennjocher/PycharmProjects/', '../')
        with open(label_path.replace('5k', '5k_bmp'), 'w') as file:
            file.write(lines)

# from utils.datasets import *; imagelist2folder()
def imagelist2folder(path='data/coco_64img.txt'):  
    # Copies all the images in a text file (list of images) into a folder
    create_folder(path[:-4])
    with open(path, 'r') as f:
        for line in f.read().splitlines():
            os.system('cp "%s" %s' % (line, path[:-4]))
            print(line)

def create_folder(path='./new_folder'):
    # Create folder
    if os.path.exists(path):
        shutil.rmtree(path)  # delete output folder
    os.makedirs(path)  # make new output folder
