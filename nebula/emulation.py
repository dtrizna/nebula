import os
import json
import logging

import speakeasy
from pefile import PEFormatError
from unicorn import UcError

from pathlib import Path
from numpy import sum
from time import time

EXCEPTIONS = (PEFormatError, UcError, IndexError, speakeasy.errors.NotSupportedError, speakeasy.errors.SpeakeasyError)

def write_error(errfile):
    # just creating an empty file to incdicate failure
    Path(errfile).touch()


def emulate(file=None, data=None, report_folder=".", i=0, l=0,
            speakeasy_config = "", forceEmulation=False):
    
    if file is None and data is None:
        raise ValueError("Either 'file' or 'data' must be specified.")
    if file:
        if not os.path.exists(file):
            raise ValueError(f"File {file} does not exist.")
        sampleName = os.path.basename(file)
        fileBase = os.path.join(report_folder, sampleName)

        analyze = False
        if not forceEmulation and os.path.exists(fileBase+".json"):
            logging.warning(f" [!] {i}/{l} Exists, skipping analysis: {fileBase}.json\n")
        elif not forceEmulation and os.path.exists(fileBase+".err"):
            logging.warning(f" [!] {i}/{l} Exists, skipping analysis: {fileBase}\n")
        else:
            analyze = True
    if data:
        fileBase = os.path.join(report_folder, int(time()))
    if not analyze:
        return None
    else:
        config = json.load(open(speakeasy_config)) if speakeasy_config else None
        se = speakeasy.Speakeasy(config=config)
        success = False
        try:
            if file:
                module = se.load_module(path=file)
            if data:
                module = se.load_module(data=data)
            se.run_module(module)
            report = se.get_report()

            took = report["emulation_total_runtime"]
            api_seq_len = sum([len(x["apis"]) for x in report["entry_points"]])
            
            if api_seq_len >= 1:
                # 1 API call is sometimes already enough
                with open(f"{fileBase}.json".replace(".exe",""), "w") as f:
                    json.dump(report["entry_points"], f, indent=4)
                success = True
        
            else:
                err = [x['error']['type'] if "error" in x.keys() and "type" in x["error"].keys() else "" for x in report['entry_points']]
                if "unsupported_api" in err:
                    try:
                        err.extend([x['error']['api']['name'] for x in report['entry_points']])
                    except KeyError:
                        err.extend([x['error']['api_name'] for x in report['entry_points']])
                logging.debug(f" [D] {i}/{l} API nr.: {api_seq_len}; Err: {err};")
                write_error(fileBase+".err")

            logging.warning(f" [+] {i}/{l} Finished emulation {file}, took: {took:.2f}s, API calls acquired: {api_seq_len}")
            return success
        except EXCEPTIONS as ex:
            logging.error(f" [-] {i}/{l} Failed emulation of {file}\nException: {ex}\n")
            write_error(fileBase+".err")
            return success
        except Exception as ex:
            logging.error(f" [-] {i}/{l} Failed emulation, general Exception: {file}\n{ex}\n")
            write_error(fileBase+".err")
            return success