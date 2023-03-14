import os
import json
from pandas import json_normalize
from collections.abc import Iterable

def flatten(l):
    for el in l:
        if isinstance(el, Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el

def report_to_apiseq(report):
    """Function parses JSON reports provided by emulaton/emulate_samples.py
    which generates files in a format of /path/to/report/<hash>.json

    Args:
        report(str, dict, list): Report in memory or fullpath of a reportfile.

    Returns:
        apiseq: sequence of API calls from emulation report.
    """
    try:
        if isinstance(report, (dict, list)):
            report = report
        elif isinstance(report, str):
            report = json.loads(report)
        elif os.path.exists(report):
            with open(report) as f:
                report = json.load(f)
    except json.decoder.JSONDecodeError as ex:
        print(f"[-] {report} JSONDecode exception: {ex}")
        return None

    report_fulldf = json_normalize(report)
    apiseq = list(flatten(report_fulldf["apis"].apply(lambda x: [y["api_name"].lower() for y in x]).values))
    return apiseq
