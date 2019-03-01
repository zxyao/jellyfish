from PIL import Image
import os
import argparse
import scipy.io

parser = argparse.ArgumentParser("mat2jpg")

parser.add_argument('--folder-prefix', type=str, required=True)
parser.add_argument('--input-dir', type=str, default='mat')
parser.add_argument('--output-dir', type=str, default='edge')
args = parser.parse_args()

MAT_PATH = args.folder_prefix + args.input_dir
OUTPUT_PATH = args.folder_prefix + args.output_dir

def mat2jpg(fname):
    mat = scipy.io.loadmat(os.path.join(MAT_PATH, fname))['predict']
    im = Image.fromarray(mat*256.)
    im = im.convert('RGB')
    im.save(os.path.join(OUTPUT_PATH, fname[:-4]+'.jpg'))

images = filter(lambda f: not f.startswith("."), os.listdir(MAT_PATH))
if not os.path.exists(OUTPUT_PATH):
    print('Created output directory {}'.format(OUTPUT_PATH))
    os.makedirs(OUTPUT_PATH)
print("Convert {:d} images from mat to jpg".format(len(images)))

for i in images:
    mat2jpg(i)