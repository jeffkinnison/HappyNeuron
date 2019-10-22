import argparse
#from .utils import mpi_read
#from .utils import mpi_map
import utils

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--input', default=None, type=str)
  parser.add_argument('--output', default=None, type=str)
  parser.add_argument('--factor', default=2, type=int)
  args = parser.parse_args()

  kv = utils.mpi_read(args.input)
  kv = utils.mpi_map(kv, utils.invert) 
  kv = utils.mpi_map(kv, utils.reduce_size) 
  utils.mpi_write(kv, args.output) 

if __name__ == '__main__':
  main()
