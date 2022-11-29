import argparse

from resizer import ImageResizer


def parse():
    """Parse command line."""
    parser = argparse.ArgumentParser('Effective resize of the binary image')
    parser.add_argument('input_img', type=str, help='Input image to be resized.')
    parser.add_argument('img_w_h', type=int, nargs='+', help='Size of the image after resize')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse()
    image_resizer = ImageResizer(
        img_filename=args.input_img,
        new_shape=args.img_w_h,
    )
    image_resizer.resize()
