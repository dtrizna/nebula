import os
import json
import sys
sys.path.extend(['.','../..'])
from nebula.constants import *
from nebula.preprocessing import JSONParser, PEDynamicFeatureExtractor

repoRoot = r"C:\Users\dtrizna\Code\nebula"

entryPointEvent = os.path.join(repoRoot, r"emulation\reportSample_EntryPoint_ransomware.json")
fullReportEvent = os.path.join(repoRoot, "emulation", "reportSample_full.json")
auditdEvent = os.path.join(repoRoot, r"data\data_raw\auditd_msft_raw_benign\1662470678.json")
events = dict(zip(["entry", "full", "auditd"], [entryPointEvent, fullReportEvent, auditdEvent]))
eventData = {}
for event in events:
    with open(events[event], 'r') as f:
        data = json.load(f)
        eventData[event] = data

# Parse the data

# parser = JSONParser(
#     fields=SPEAKEASY_RECORD_SUBFILTER_OPTIMAL_FIELDS
# )
# blah2 = parser.filter(eventData["entry"])
# print(blah2)

extractor = PEDynamicFeatureExtractor()
jsonEntry = extractor.filter_and_normalize_report(eventData["entry"])
# with open("./test.json", 'w') as f:
#     json.dump(jsonEntry, f, indent=4)

parser = JSONParser(
    fields=AUDITD_FIELDS,
    normalized=True
)
auditdEvents = eventData["auditd"][0:2]
print(auditdEvents)
blah = parser.filter(auditdEvents)
print(blah)
