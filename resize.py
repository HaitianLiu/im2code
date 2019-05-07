import argparse
import os
from PIL import Image
import sys
import shutil

def resize_image(image, size):
    """Resize an image to the given size."""
    return image.resize(size, Image.ANTIALIAS)


def resize_images(image_dir, output_dir, size):
    """Resize the images in 'image_dir' and save into 'output_dir'."""
    print("Ready to serve.")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dirs = os.listdir(image_dir)

    for d in dirs:
        os.makedirs(os.path.join(output_dir, d))
        file_names = os.listdir(os.path.join(image_dir, d))
        num_images = len(file_names) // 2
        for i, file_name in enumerate(file_names):
            if file_name.find('.png') != -1:
                with open(os.path.join(image_dir, d, file_name), 'r+b') as f:
                    with Image.open(f) as img:
                        img = resize_image(img, size)
                        img.save(os.path.join(output_dir, d, file_name))
            elif file_name.find('.gui') != -1:
                shutil.copy2(os.path.join(image_dir, d, file_name), os.path.join(output_dir, d, file_name))
            if (i+1) % 150 == 0 or (i+1) == num_images * 2:
                print("[{}/{}] Resized the {} images and saved into '{}'."
                    .format((i+1) // 2, num_images, d, output_dir + d))
    print("Job's done.")

def main(args):
    image_dir = args.image_dir
    output_dir = args.output_dir
    image_size = [args.image_size, args.image_size]
    resize_images(image_dir, output_dir, image_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default='./data/raw/',
                        help='directory for train images')
    parser.add_argument('--output_dir', type=str, default='./data/resized/',
                        help='directory for saving resized images')
    parser.add_argument('--image_size', type=int, default=256,
                        help='size for image after processing')
    args = parser.parse_args()
    main(args)
