import os
import string
import heapq
import numpy as np
from pprint import pprint

path = '/root/xcn/file4lab/esoda/datasets/'
root_folders = os.listdir(path)
test_folder = 'conf_chi'
test_file = 'conf_chi_Densmore12.txt'


for folder in root_folders:

    files = os.listdir(path + folder)
    for file in files:
        print(file)
        # f = open(path + folder + '/' + file)
        # iter_f = iter(f)
