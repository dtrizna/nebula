import os
import json
import logging

import speakeasy
from pefile import PEFormatError
from unicorn import UcError

from pathlib import Path
from numpy import sum


def write_error(errfile):
    # creating of empty file 
    Path(errfile).touch()


def emulate(file, report_folder, i=0, l=0,
            speakeasy_config = ""):
    samplename = os.path.basename(file)
    reportfile = f"{report_folder}/{samplename}.json"
    errfile = f"{report_folder}/{samplename}.err"

    success = False
    if os.path.exists(reportfile):
        logging.warning(f" [!] {i}/{l} Exists, skipping analysis: {reportfile}\n")
    elif os.path.exists(errfile):
        logging.warning(f" [!] {i}/{l} Exists, skipping analysis: {errfile}\n")
    else:
        try:
            config = json.load(open(speakeasy_config)) if speakeasy_config else None
            se = speakeasy.Speakeasy(config=config)
            module = se.load_module(file)
            se.run_module(module)
            report = se.get_report()

            took = report["emulation_total_runtime"]
            api_seq_len = sum([len(x["apis"]) for x in report["entry_points"]])
            
            if api_seq_len >= 1:
                # 1 API call is sometimes already enough
                with open(f"{reportfile}".replace(".exe",""), "w") as f:
                    json.dump(report["entry_points"], f, indent=4)
                success = True
        
            if api_seq_len == 1 or api_seq_len == 0:
                # some uninformative failures with 0 or 1 API calls - e.g. ordinal_100
                api = [x['apis'] for x in report['entry_points']][0] if api_seq_len == 1 else ''
                err = [x['error']['type'] if "error" in x.keys() and "type" in x["error"].keys() else "" for x in report['entry_points']]
                if "unsupported_api" in err:
                    try:
                        err.extend([x['error']['api']['name'] for x in report['entry_points']])
                    except KeyError:
                        err.extend([x['error']['api_name'] for x in report['entry_points']])
                logging.debug(f" [D] {i}/{l} API nr.: {api_seq_len}; Err: {err}; APIs: {api}")
                if api_seq_len == 0:
                    write_error(errfile)

            logging.warning(f" [+] {i}/{l} Finished emulation {file}, took: {took:.2f}s, API calls acquired: {api_seq_len}")
            return success

        except PEFormatError as ex:
            logging.error(f" [-] {i}/{l} Failed emulation, PEFormatError: {file}\n{ex}\n")
            write_error(errfile)
            return success
        except UcError as ex:
            logging.error(f" [-] {i}/{l} Failed emulation, UcError: {file}\n{ex}\n")
            write_error(errfile)
            return success
        except IndexError as ex:
            logging.error(f" [-] {i}/{l} Failed emulation, IndexError: {file}\n{ex}\n")
            write_error(errfile)
            return success
        except speakeasy.errors.NotSupportedError as ex:
            logging.error(f" [-] {i}/{l} Failed emulation, NotSupportedError: {file}\n{ex}\n")
            write_error(errfile)
            return success
        except speakeasy.errors.SpeakeasyError as ex:
            logging.error(f" [-] {i}/{l} Failed emulation, SpeakEasyError: {file}\n{ex}\n")
            write_error(errfile)
            return success
        except Exception as ex:
            logging.error(f" [-] {i}/{l} Failed emulation, general Exception: {file}\n{ex}\n")
            write_error(errfile)
            return success