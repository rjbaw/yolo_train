import os
import random
import shutil
import argparse

def random_proportion(path, train_path, val_path, proportion):
    def init():
        if not os.path.exists(os.path.join(path,train_path)):
            os.mkdir(os.path.join(path,train_path))
        if not os.path.exists(os.path.join(path,val_path)):
            os.mkdir(os.path.join(path,val_path))
    def move_files(new_path, choice):
        base = os.path.splitext(choice)[0]
        shutil.move(os.path.join(path,choice),
                    os.path.join(path,new_path,choice))
        shutil.move(os.path.join(path,base+XML_EXT),
                    os.path.join(path,new_path,base+XML_EXT))
        
    init()
    files = [f for f in os.listdir(path) \
             if ((os.path.splitext(f)[-1] != XML_EXT) and (f != train_path) and (f != val_path)) ]
    print(len(files))
    n = 0
    while len(files) != 0:
        choice = random.choice(files)
        if random.random() < proportion:
            move_files(train_path, choice)
            n += 1
        else:
            move_files(val_path, choice)
        files.remove(choice)
    print(n)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='egg_tray',
                        help='dataset path')
    parser.add_argument('--proportion', type=int, default='0.8',
                        help='Proportion of train/total')
    arg = parser.parse_args()
    XML_EXT = '.xml'
    train_path = 'train'
    val_path = 'val'
    random_proportion(arg.data, train_path, val_path, arg.proportion)
