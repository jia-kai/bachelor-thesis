#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# $File: main.py
# $Date: Sun Feb 22 18:22:47 2015 +0800
# $Author: jiakai <jia.kai66@gmail.com>

from nasmia.io import ScenePackReader
from nasmia.visualize import view_3d_data

import argparse

def main():
    parser = argparse.ArgumentParser(
        description='demo for reading images',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('img_fpath')
    args = parser.parse_args()

    reader = ScenePackReader(args.img_fpath)
    view_3d_data(reader.scenes[0].objects[0].data)

if __name__ == '__main__':
    main()
