from argparse import ArgumentParser
from datatools import vis_2Ddata


parser = ArgumentParser()
parser.add_argument("data_dir", type=str)
vis_2Ddata(parser.parse_args())
