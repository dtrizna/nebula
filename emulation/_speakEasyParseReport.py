import sys
import json
sys.path.append("../")
from nebula.preprocessing import PEDynamicFeatureExtractor

report = "./reportSample_EntryPoint_ransomware.json"
with open(report, "r") as f:
    report = json.load(f)

extractor = PEDynamicFeatureExtractor(
    speakeasyConfig="./_speakeasyConfig.json"
)
parsedJson = extractor.filter_and_normalize_report(report)

with open("./parsedJson.json", "w") as f:
    f.write(parsedJson)
