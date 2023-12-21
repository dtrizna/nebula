import sys
import json
sys.path.append("../")
from nebula.preprocessing import PEDynamicFeatureExtractor

report = "./report_example_entrypoint_ransomware.json"
with open(report, "r") as f:
    report = json.load(f)

extractor = PEDynamicFeatureExtractor(
    speakeasyConfig="./speakeasy_config.json"
)
parsedJson = extractor.filter_and_normalize_report(report)

with open("./parsed_json.json", "w") as f:
    f.write(parsedJson)
