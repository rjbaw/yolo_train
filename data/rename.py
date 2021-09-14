import os
import uuid
import argparse

class rename_rand_pair:
    def __init__(self, save_dir, label_suffix):
        self.save_dir = save_dir
        self.label_suffix = label_suffix
        self.img_suffix = ['.jpg', '.tiff']
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
#        for n in save:
#            names.remove(n)
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='egg_tray',
                        help='dataset path')
    arg = parser.parse_args()
    label_suffix = '.xml'
    x = rename_rand_pair(arg.data, label_suffix)
    x.main()
