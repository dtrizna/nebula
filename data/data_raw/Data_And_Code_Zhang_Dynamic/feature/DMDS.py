#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import os
import re
import json
import hashlib
import logging
from sklearn.feature_extraction import FeatureHasher


class FeatureType(object):
    ''' Base class from which each feature type may inherit '''

    name = ''
    dim = 0

    def __repr__(self):
        return '{}({})'.format(self.name, self.dim)

    def raw_features(self, input_dict):
        ''' Generate a JSON-able representation of the file '''
        raise (NotImplemented)

    def process_features(self, raw_obj):
        ''' Generate a feature vector from the raw features '''
        raise (NotImplemented)

    def feature_vector(self, input_dict):
        ''' Directly calculate the feature vector from the sample itself. This should only be implemented differently
        if there are significant speedups to be gained from combining the two functions. '''
        return self.process_raw_features(self.raw_features(input_dict))


class APIName(FeatureType):
    ''' api_name hash info '''

    name = 'api_name'
    dim = 8

    def __init__(self):
        super(FeatureType, self).__init__()
        self._name = re.compile('^[a-z]+|[A-Z][^A-Z]*')

    def raw_features(self, input_dict):
        """
        input_dict: string
        """
        tmp = self._name.findall(input_dict)
        hasher = FeatureHasher(self.dim, input_type="string").transform([tmp]).toarray()[0]
        return hasher

    def process_raw_features(self, raw_obj):
        return raw_obj


class APICategory(FeatureType):
    ''' api_category hash info '''
    
    name = 'api_category'
    dim = 4

    def __init__(self):
        super(FeatureType, self).__init__()

    def raw_features(self, input_dict):
        hasher = FeatureHasher(self.dim, input_type="string").transform([input_dict]).toarray()[0]
        return hasher

    def process_raw_features(self, raw_obj):
        return raw_obj


class IntInfo(FeatureType):
    ''' int hash info '''

    name = 'int'
    dim = 16

    def __init__(self):
        super(FeatureType, self).__init__()

    def raw_features(self, input_dict):
        hasher = FeatureHasher(self.dim).transform([input_dict]).toarray()[0]
        return hasher

    def process_raw_features(self, raw_obj):
        return raw_obj


class PRUIInfo(FeatureType):
    ''' Path, Registry, Urls, IPs hash info '''

    name = 'prui'
    dim = 16 + 8 + 12 + 16 + 12

    def __init__(self):
        super(FeatureType, self).__init__()
        self._paths = re.compile('^c:\\\\', re.IGNORECASE)
        self._dlls = re.compile('.+\.dll$', re.IGNORECASE)
        self._urls = re.compile('^https?://(.+?)[/|\s|:]', re.IGNORECASE)
        self._registry = re.compile('^HKEY_')
        self._ips = re.compile('^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}')

    def raw_features(self, input_dict):
        paths = np.zeros((16,), dtype=np.float32)
        dlls = np.zeros((8,), dtype=np.float32)
        registry = np.zeros((12,), dtype=np.float32)
        urls = np.zeros((16,), dtype=np.float32)
        ips = np.zeros((12,), dtype=np.float32)
        for str_name, str_value in input_dict.items():
            if self._dlls.match(str_value):
                tmp = re.split('//|\\\\|\.', str_value)[:-1]
                tmp = ['\\'.join(tmp[:i]) for i in range(1, len(tmp) + 1)]
                dlls += FeatureHasher(8, input_type="string").transform([tmp]).toarray()[0]
            if self._paths.match(str_value):
                tmp = re.split('//|\\\\|\.', str_value)[:-1]
                tmp = ['\\'.join(tmp[:i]) for i in range(1, len(tmp) + 1)]
                paths += FeatureHasher(16, input_type="string").transform([tmp]).toarray()[0]
            elif self._registry.match(str_value):
                tmp = str_value.split('\\')[:6]
                tmp = ['\\'.join(tmp[:i]) for i in range(1, len(tmp) + 1)]
                registry += FeatureHasher(12, input_type="string").transform([tmp]).toarray()[0]
            elif self._urls.match(str_value):
                tmp = self._urls.split(str_value + "/")[1]
                tmp = tmp.split('.')[::-1]
                tmp = ['.'.join(tmp[:i][::-1]) for i in range(1, len(tmp) + 1)]
                urls += FeatureHasher(16, input_type="string").transform([tmp]).toarray()[0]
            elif self._ips.match(str_value):
                tmp = str_value.split('.')
                tmp = ['.'.join(tmp[:i]) for i in range(1, len(tmp) + 1)]
                ips += FeatureHasher(12, input_type="string").transform([tmp]).toarray()[0]
        return np.hstack([paths, dlls, registry, urls, ips]).astype(np.float32)

    def process_raw_features(self, raw_obj):
        return raw_obj


class StringsInfo(FeatureType):
    ''' Other printable strings hash info '''

    name = 'strings'
    dim = 8

    def __init__(self):
        super(FeatureType, self).__init__()
        self._allstrings = re.compile(b'[\x20-\x7f]{5,}')
        self._paths = re.compile(b'c:\\\\', re.IGNORECASE)
        self._dlls = re.compile(b'\\.dll', re.IGNORECASE)
        self._urls = re.compile(b'https?://', re.IGNORECASE)
        self._registry = re.compile(b'HKEY_')
        self._mz = re.compile(b'MZ')
        self._ips = re.compile(b'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}')
        super(FeatureType, self).__init__()

    def raw_features(self, input_dict):
        bytez = '\x11'.join(input_dict.values()).encode('UTF-8', 'ignore')
        allstrings = self._allstrings.findall(bytez)
        if allstrings:
            # statistics about strings:
            string_lengths = [len(s) for s in allstrings]
            avlength = sum(string_lengths) / len(string_lengths)
            # map printable characters 0x20 - 0x7f to an int array consisting of 0-95, inclusive
            as_shifted_string = [b - ord(b'\x20') for b in b''.join(allstrings)]
            c = np.bincount(as_shifted_string, minlength=96)  # histogram count
            # distribution of characters in printable strings
            csum = c.sum()
            p = c.astype(np.float32) / csum
            wh = np.where(c)[0]
            H = np.sum(-p[wh] * np.log2(p[wh]))  # entropy
        else:
            avlength = 0
            c = np.zeros((96,), dtype=np.float32)
            H = 0
            csum = 0
        return {
            'numstrings': len(allstrings),
            'avlength': avlength,
            'printables': int(csum),
            'entropy': float(H),
            'paths': len(self._paths.findall(bytez)),
            'dlls': len(self._dlls.findall(bytez)),
            'urls': len(self._urls.findall(bytez)),
            'registry': len(self._registry.findall(bytez)),
            'ips': len(self._ips.findall(bytez)),
            'MZ': len(self._mz.findall(bytez))
        }

    def process_raw_features(self, raw_obj):
        return np.hstack([
            raw_obj['numstrings'], raw_obj['avlength'], raw_obj['printables'],
            raw_obj['entropy'], raw_obj['paths'], raw_obj['dlls'], raw_obj['urls'],
            raw_obj['registry'], raw_obj['ips'], raw_obj['MZ']
        ]).astype(np.float32)


class DMDS(object):

    def __init__(self, file_name, input_path, output_path, max_len, idx):
        logging.info('Generating vector for task %s' % idx)

        self.idx = idx
        self.behaviour_report = None
        self.nrdma_output = None
        self.max_len = max_len
        self.features = dict((fe.name, fe) for fe in
                             [APIName(), APICategory(), IntInfo(), PRUIInfo(), StringsInfo()])
        self.input_path = input_path
        self.output_path = output_path
        self.file_name = file_name
        self.infile = self.input_path.format(self.file_name)
        self.outfile = self.output_path.format(self.file_name)

    def parse(self):
        if not os.path.exists(self.infile):
            logging.warning("Behaviour report does not exist.")
            return False
        if os.path.exists(self.outfile):
            logging.warning("Behaviour report already parsed.")
            return False
        try:
            json_data = open(self.infile, "r")
            self.behaviour_report = json.load(json_data)
            return True
        except Exception as e:
            logging.error('Could not parse the behaviour report. {%s}' % e)
            return False

    def write(self):
        outputfile = self.output_path.format(self.file_name)
        logging.info("Writing task %s report to: %s" % (self.idx, outputfile))
        np.save(outputfile, self.nrdma_output)
        return True

    def add_to_output(self, sample):
        if self.nrdma_output is None:
            self.nrdma_output = [sample]
        else:
            self.nrdma_output = np.append(self.nrdma_output, [sample], axis=0)
        return len(self.nrdma_output)

    def convert_thread(self, pid, tid, api_calls):
        previous_hashed = ""
        for call in api_calls:
            if self.nrdma_output is not None and len(self.nrdma_output) >= self.max_len:
                return True
            if 'api' not in call:
                continue
            if call['api'][:2] == '__':
                continue
            if 'arguments' not in call:
                call['arguments'] = {}
            if 'category' not in call:
                call['category'] = ""
            if 'status' not in call:
                call['status'] = 0
            arguments = call['arguments']
            category = call['category']
            api = call['api']
            call_sign = api + "-" + str(arguments)
            current_hashed = hashlib.md5(call_sign.encode()).hexdigest()
            if previous_hashed == current_hashed:
                continue
            else:
                previous_hashed = current_hashed
            api_name_hashed = self.features['api_name'].feature_vector(api)
            api_category_hashed = self.features['api_category'].feature_vector(
                category)
            api_int_dict, api_str_dict = {}, {}
            for c_n, c_v in arguments.items():
                if isinstance(c_v, (list, dict, tuple)):
                    continue
                if isinstance(c_v, (int, float)):
                    api_int_dict[c_n] = np.log(np.abs(c_v) + 1)
                else:
                    if c_v[:2] == '0x':
                        continue
                    api_str_dict[c_n] = c_v
            try:
                api_int_hashed = self.features['int'].feature_vector(api_int_dict)
                api_prui_hashed = self.features['prui'].feature_vector(
                    api_str_dict)
                api_str_hashed = self.features['strings'].feature_vector(
                    api_str_dict)
                hashed_feature = np.hstack(
                    [api_name_hashed, api_category_hashed, api_int_hashed, api_prui_hashed, api_str_hashed]).astype(
                    np.float32)
                self.add_to_output(hashed_feature)
            except Exception as e:
                logging.error("Task %s error: %s" % (self.idx, e))
                pass

        return True

    #  Launch the conversion on all threads in the JSON
    def convert(self):
        processes = {}
        try:
            procs = self.behaviour_report['processes']
            for proc in procs:
                process_id = proc['pid']
                parent_id = proc['ppid']
                process_name = proc['process_name']
                calls = proc['calls']
                #  Create a dictionnary of threads
                # The key is the n° of the thread
                # The content is all calls he makes
                threads = {}
                for call in calls:
                    thread_id = call['tid']
                    try:
                        threads[thread_id].append(call)
                    except:
                        threads[thread_id] = []
                        threads[thread_id].append(call)

                # Create a dictionnary of process
                # The key is the id of the process
                processes[process_id] = {}
                processes[process_id]["parent_id"] = parent_id
                processes[process_id]["process_name"] = process_name
                processes[process_id]["threads"] = threads
        except Exception as e:
            logging.error("Task %s error: %s" % (self.idx, e))
        # For all processes...
        for p_id in processes:
            #  For each threads of those processes...
            for t_id in processes[p_id]["threads"]:
                # Convert the thread
                self.convert_thread(p_id, t_id, processes[p_id]["threads"][t_id])
        return True


if __name__ == '__main__':
    #file_name = '8FFCD330EDAEFAC6320E9D00D39EDBE2D396BCC5FE4D55EFC1D5082B78E9500D_180252_1'
    file_name = r"data\data_raw\Data_And_Code_Zhang_Dynamic\feature\8FFCD330EDAEFAC6320E9D00D39EDBE2D396BCC5FE4D55EFC1D5082B78E9500D_180252_1"
    input_path = f"./{file_name}.json"
    output_path = f"./{file_name}.npy"
    dmds = DMDS(file_name, input_path, output_path, 1000, 40)
    dmds.parse()
    dmds.convert()
    dmds.write()
