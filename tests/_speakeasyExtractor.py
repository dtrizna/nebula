import sys
sys.path.extend([".", ".."])
from nebula import PEDynamicFeatureExtractor

path = r"C:\windows\syswow64\wusa.exe"

with open(path, "rb") as f:
    bytez = f.read()

extractor = PEDynamicFeatureExtractor()

jsonP = extractor.emulate(path=path)
jsonB = extractor.emulate(data=bytez)
import pdb; pdb.set_trace()