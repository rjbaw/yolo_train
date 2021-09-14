from models import *  
from utils.datasets import *
from utils.utils import *
from utils.xml_generator import PascalVocWriter
import numpy as np
import time
import cv2
import torch
import torchvision
import argparse

IMG_EXT = '.jpg'
XML_EXT = '.xml'

class Detector:
    def __init__(self, opt):
        self.out = opt.output
        self.half = opt.half
        self.view_img = opt.view_img
        self.cfg = opt.cfg
        self.conf_thres = opt.conf_thres
        self.nms_thres = opt.conf_thres
        self.fourcc = opt.fourcc
        self.save_img = opt.save
        self.generate_labels = opt.save_labels
        self.device = torch_utils.select_device(opt.device)
        self.classes = load_classes(parse_data_cfg(opt.data)['names'])
        self.clscolors = self.get_colors()
        self.vid_path = None
        self.vid_writer = None
        self.t0 = time.time()
        self.crop = False
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def initialize_folder(self, dirs):
        if os.path.exists(dirs):
            shutil.rmtree(dirs)  
        os.makedirs(dirs)  

    def get_sources(self, source):
        keywords = ['rtsp', 'http', '.txt', 'v4l2src']
        try:
            int(source)
            webcam = True
        except Exception as e:
            webcam = np.any([True if (key in source) else False for key in keywords])
        if webcam:
            torch.backends.cudnn.benchmark = True  
            dataset = LiveFeed(source)
        else:
            dataset = MediaFiles(source)
        return dataset

    def load_model(self, weights):
        model = Darknet(self.cfg, self.img_size)
        if weights.endswith('.pt'): 
            model.load_state_dict(torch.load(weights, map_location=self.device)['model'])
        else:  
            _ = load_darknet_weights(model, weights)
        model.to(self.device).eval()
        self.half = self.half and self.device.type != 'cpu'  
        if self.half:
            model.half()
        return model

    def get_colors(self):
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.classes))]
        return colors

    def yolov3_transform(self, im0s):
        if type(im0s) == list:
            img = [letterbox(x, new_shape=self.img_size, interp=cv2.INTER_LINEAR)[0] for x in im0s]
            img = np.stack(img, 0)
            img = img[:, :, :, ::-1].transpose(0, 3, 1, 2) 
            img = np.ascontiguousarray(img, dtype=np.float16 if self.half else np.float32)  
            img /= 255.0  
        else:
            img = letterbox(im0s, new_shape=self.img_size)[0]
            img = img[:, :, ::-1].transpose(2, 0, 1)
            img = np.ascontiguousarray(img, dtype=np.float16 if self.half else np.float32)  
            img /= 255.0 
        if img.ndim == 3:
            img = img.unsqueeze(0)
        img = torch.from_numpy(img).to(self.device)
        #if img.ndimension() == 3:
        #    img = img.unsqueeze(0)
        return img

    def detect(self, path, im0s, vid_cap, save_txt=False):
        def eval_pred(i, det):
            if det != None and len(det):
                det[:,:4] = scale_coords(img.shape[2:], det[:, :4], im0s[i].shape).round()
                rects = det.cpu().numpy()
            self.process_results(rects, im0s[i], path[i], vid_cap)
        img = self.yolov3_transform(im0s)
        pred = self.model(img)[0]
        if self.half:
            pred = pred.float()
        pred = non_max_suppression(pred, self.conf_thres, self.nms_thres)
        with ThreadPoolExecutor() as executor:
            all( executor.map(eval_pred, range(len(pred)), pred) )

    def save_crop_img(self, crop_imgs):
        if len(crop_imgs) > 1:
            for crop_img in crop_imgs:
                i = 1
                image_dir = os.path.join(self.out, str(i) + IMG_EXT)
                while os.path.exists(image_dir):
                    i += 1
                    image_dir = os.path.join(self.out, str(i) + IMG_EXT)
                try:
                    cv2.imwrite(image_dir, crop_img)
                except:
                    pass

    def save_video(self, im0, vid_cap, save_path):
        if self.vid_path != save_path:  
            self.tack[0].reset
            self.vid_path = save_path
            if isinstance(self.vid_writer, cv2.VideoWriter):
                self.vid_writer.release()  
            fps = vid_cap.get(cv2.CAP_PROP_FPS)
            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.vid_writer = cv2.VideoWriter(save_path,
                                              cv2.VideoWriter_fourcc(*self.fourcc), # *'MPEG'
                                              30, (w, h))
        self.vid_writer.write(im0)

    def save_to_out(self, im0, save_path, vid_cap, crop_img=[]):
        if self.dataset.mode == 'images':
            cv2.imwrite(save_path, im0)
        else:
            self.save_video(im0, vid_cap, save_path)
        if self.crop:
            self.save_crop_img(crop_img)

    def process_results(self, rects, im0, path, vid_cap):
        def plot_bbox(rect, im0):
            if self.crop:
                crop_imgs.append(
                    im0[int(rect[1]):int(rect[3]),\
                        int(rect[0]):int(rect[2])]
                )
            if self.save_img or self.view_img:  
                label = '%s %.2f' % (self.classes[int(rect[6])], rect[5])
                try: 
                    im0 = plot_one_box(rect[:4], im0,
                                       label=label, color=self.clscolors[int(rect[6])])
                except Exception as e:
                    print(e)
        crop_imgs = []
        save_path = str(Path(self.out) / Path(path).name)
        im0_ori = copy.copy(im0)
        for rect in rects:
            plot_bbox(rect, im0)
        if self.view_img:
            cv2.imshow(path, cv2.resize(im0,(int(1080/im0.shape[0]*im0.shape[1]),1080)))
            #cv2.imshow(path, im0)
        if self.save_img:
            self.save_to_out(im0_ori, save_path, vid_cap, crop_imgs)
        if self.generate_labels:
            self.save_xml_labels(im0_ori, save_path, rects, im0.shape)

    def save_xml_labels(self, im0, save_path, rects, shape):
        if (len(rects) == 0) or (np.sum(rects[:,5]) == 0):
            return
        height, width, depth = shape
        depth = 3
        imageshape = [height, width, depth]
        if self.dataset.mode != 'images':
            i = 1
            image_dir = os.path.join(self.out, str(i) + IMG_EXT)
            while os.path.exists(image_dir):
                i += 1
                image_dir = os.path.join(self.out, str(i) + IMG_EXT)
            try:
                cv2.imwrite(image_dir, im0)
            except:
                pass
            filename = str(i)
            xml_path = os.path.join(self.out, str(i) + XML_EXT)
        else:
            filename = os.path.basename(save_path)
            xml_path = os.path.join(os.path.dirname(save_path),
                                    filename.split('.')[0] + XML_EXT)
        writer = PascalVocWriter(self.out, filename, imageshape, localImgPath=xml_path)
        for rect in rects:
            label = self.classes[int(rect[-1])]
            difficult = 0
            writer.addBndBox(rect[0], rect[1], rect[2],
                             rect[3], label, difficult)
        writer.save(targetFile=xml_path)

    def main(self, save_txt = False):
        self.initialize_folder(self.out)
        self.dataset = self.get_sources(opt.source)
        self.img_size = opt.img_size  # (320, 192) or (416, 256) or (608, 352) [(height, width)]
        self.model = self.load_model(opt.weights)
        for path, im0s, vid_cap in self.dataset:
            t = time.time()
            self.detect(path, im0s, vid_cap)
            if cv2.waitKey(1) & 0xFF == ord('q'): 
                break
        print("elapsed time: {:.2f}".format(time.time() - self.t0))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp.cfg',
                        help='cfg file path')
    parser.add_argument('--data', type=str, default='data/coco.data',
                        help='data file path')
    parser.add_argument('--weights', type=str, default='weights/yolov3-spp.weights',
                        help='path to weights file')
    parser.add_argument('--source', type=str, default='data/samples',
                        help='image sources')
    parser.add_argument('--output', type=str, default='output',
                        help='results folder')
    parser.add_argument('--img-size', type=int, default=416,
                        help='inference square length (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3,
                        help='confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.5,
                        help='iou threshold for non-maximum suppression')
    parser.add_argument('--fourcc', type=str, default='mp4v',
                        help='video output codec')
    parser.add_argument('--device', default='',
                        help='device id e.g. 0 or 0,1 or cpu')
    parser.add_argument('--half', action='store_true',
                        help='use half precision (FP16)')
    parser.add_argument('--view-img', action='store_true',
                        help='display results')
    parser.add_argument('--save', action='store_false',
                        help='save output')
    parser.add_argument('--save-labels', action='store_false',
                        help='save labels')
    opt = parser.parse_args()
    print(opt)
    with torch.no_grad():
        Detector(opt).main()
