import argparse

from resizer import ImageResizer


def parse():
    """Parse command line."""
    parser = argparse.ArgumentParser('Effective resize of the binary image.')
    parser.add_argument(
        'input_img',
        type=str,
        help='Input image to be resized.',
    )
    parser.add_argument(
        'img_w_h',
        type=int,
        nargs='+',
        help='Size of the image after resize.',
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['naive_nearest', 'vectorized_nearest', 'naive_bilinear', 'vectorized_bilinear'],
        default='vectorized_bilinear',
        help='Mode of resizing.',
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse()
    image_resizer = ImageResizer(
        img_filename=args.input_img,
        new_shape=args.img_w_h,
        mode=args.mode,
    )
    image_resizer.resize()
