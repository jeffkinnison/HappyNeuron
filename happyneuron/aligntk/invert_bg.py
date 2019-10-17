import argparse
from .utils import mpi_process

def invert_background(image, invert_first=False):
  if invert_first:
    image = 255 - image
  image[image == 255] = 0
  return image
def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('input', default=None, type=str)
  parser.add_argument('output', default=None, type=str)
  parser.add_argument('--invert_first', action="store_true")
  args = parser.parse_args()

  params = dict(invert_first=args.invert_first)

  mpi_process(args.input, args.output, invert_background, params)

if __name__ == '__main__':
  main()
