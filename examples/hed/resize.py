from PIL import Image
import os
import argparse

parser = argparse.ArgumentParser("resize images")

parser.add_argument('--input-dir', type=str, default="jellyfish")
parser.add_argument('--output-dir', type=str, default="jellyfish256")
parser.add_argument('--size', type=int, default=256)
args = parser.parse_args()

IMAGE_PATH = args.input_dir
OUTPUT_PATH = args.output_dir
size = args.size

def resize(fname):
    im = Image.open(os.path.join(IMAGE_PATH, fname))
    im = im.resize((size, size), Image.ANTIALIAS)
    if im.mode != 'RGB':
        rgbim = Image.new('RGB', im.size)
        rgbim.paste(im)
        im = rgbim
    im.save(os.path.join(OUTPUT_PATH, fname[:-4]+'.jpg'))

images = list(filter(lambda f: not f.startswith('.'), os.listdir(IMAGE_PATH)))
if not os.path.exists(OUTPUT_PATH):
    print('Created output directory {}'.format(OUTPUT_PATH))
    os.makedirs(OUTPUT_PATH)
print('Resizing {:d} images to size {:d}x{:d}'.format(len(images), size, size))

for i in images:
    resize(i)
