import argparse
import logging

import os
import time
import threading

import sys
sys.path.append(".")
sys.path.append("..\\..\\")
from nebula.preprocessing import PEDynamicFeatureExtractor

CONFIG = r"C:\Users\dtrizna\Code\nebula\preprocessing\_speakeasyConfig.json"

if __name__ == "__main__":
    # E.G. RUN AS:
    # python3 preprocessing\_speakeasyEmulateSamples.py 
    # --sample-prefix C:\Windows\SysWOW64\ 
    # --output C:\Users\dtrizna\Code\nebula\data\data_raw\windows_emulationDataset\report_windows_syswow64 
    # --logfile emulateWindowsSysWow64.log 
    # --sample-filter .exe

    parser = argparse.ArgumentParser(description="Collects tracelogs from sample emulation.")

    # data specifics
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--samples", type=str, nargs="+", help="Path to PE file(s) for emulation. Can specify multiple (separate by space).")
    group.add_argument("--sample-prefix", type=str, nargs="+", help="Prefix of path PE files for emulation. Can specify multiple (separate by space).")
    parser.add_argument("--sample-filter", type=str, default=None, help="If provided, this will be considered as filter for samples.")

    parser.add_argument("--output", type=str, default="reports", help="Directory where to store emulation reports.")
    
    parser.add_argument("--start-idx", type=int, default=0, help="If provided, emulation will start from file with this index.")
    parser.add_argument("--threads", type=int, default=5, help="If provided, emulation will start from file with this index.")

    parser.add_argument("--logfile", type=str, help="File to store logging messages")
    parser.add_argument("--debug", action="store_true", help="Provide with DEBUG level information from packages")

    args = parser.parse_args()

    # if logfile argument present - log to a file instead of stdout
    level = logging.DEBUG if args.debug else logging.WARNING
    logconfig = {
        "level": level,
        "format": "%(asctime)-15s %(name)-5s %(levelname)-8s %(message)s"
    }
    if args.logfile:
        logconfig["handlers"] = [logging.FileHandler(args.logfile, 'a', 'utf-8')]
    logging.basicConfig(**logconfig)
    
    logging.warning(f" [!] Starting emulation.\n")

    # parsing files
    filelist = []
    if args.samples:
        for file in args.samples:
            if os.path.exists(file):
                if os.path.isdir(file):
                    logging.error(f" [-] {file} is a directory... If you want parse samples within that directory, use --sample-prefix!")
                    sys.exit(1)
                filelist.append(file)
            else:
                logging.error(f" [-] {file} does not exist...")
    elif args.sample_prefix:
        for prefix in args.sample_prefix:
            prefix = prefix.rstrip("/")
            if os.path.exists(prefix):
                # folder
                filelist.extend([os.path.join(prefix,x) for x in os.listdir(prefix)])
            else:
                folder = "/".join(prefix.split("/")[:-1])
                if os.path.exists(folder):
                    prefix = prefix.split("/")[-1]
                    files = [os.path.join(folder,x) for x in os.listdir(folder) if x.startswith(prefix)]
                    filelist.extend(files)
                else:
                    logging.error(f" [-] {prefix} folder does not exist...")
    else:
        logging.error(" [-] Specify either --samples or --sample-prefix.")
        sys.exit(1)
    if args.sample_filter:
        filelist = [x for x in filelist if args.sample_filter in x]
    
    # emulate samples
    #timestamp = int(time.time())
    report_folder = f"{args.output}"
    os.makedirs(report_folder, exist_ok=True)

    l = len(filelist)
    
    extractor = PEDynamicFeatureExtractor(
        speakeasyConfig="./_speakeasyConfig.json"
    )
    for i,file in enumerate(filelist):
        print(f" [!] Emulating {i}\{l}", end="\r")
        if i < args.start_idx:
            continue
        while len(threading.enumerate()) > args.threads:
            time.sleep(0.1)

        t = threading.Thread(target=extractor.emulate, args=(file, report_folder, i, l, CONFIG))
        t.start()
        logging.debug(f" [D] Started theread: {i}\{l}")