#!/usr/bin/env python
# encoding: utf-8

import logging
import os
from Cuckoo2DMDS.DMDS import DMDS
from multiprocessing import Pool


class Cuckoo2DMDS(object):
    def __init__(self, file_name, input_path, output_path, max_len, idx):
        """
        :param file_name:
        :param input_path:
        :param output_path:
        :param max_len:
        """
        self.file_name = file_name
        self.input_path = input_path + "{0}.json"
        self.output_path = output_path + "{0}.npy"
        self.max_len = max_len
        self.idx = idx

    def run(self):
        """

        :return:
        """
        logging.info("Task %s start" % self.idx)
        dmds = DMDS(self.file_name, self.input_path, self.output_path, self.max_len, self.idx)
        if dmds.parse() and dmds.convert():
            dmds.write()
            logging.info("Task %s finish" % self.idx)


if __name__ == '__main__':
    pool_size = 10
    max_len = 1000
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG)

    input_path = "/Users/admin/Data/json/"
    output_path = "/Users/admin/Data/npy/"

    files = [f[:-5] for f in os.listdir(input_path) if f[-5:] == '.json']
    p = Pool(pool_size)
    for f in files:
        task = Cuckoo2DMDS(f, input_path, output_path, max_len, f)
        p.apply_async(task.run())
    p.close()
    p.join()
