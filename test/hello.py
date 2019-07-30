import numpy
import argparse

parser = argparse.ArgumentParser(description='face model test')
parser.add_argument('--namez', default='Talgat', help='')
args = parser.parse_args()

name = args.namez

print('Hello ', name)

