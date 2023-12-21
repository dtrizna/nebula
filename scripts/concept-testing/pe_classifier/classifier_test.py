import sys
sys.path.extend(['../..', '.'])
from nebula import PEHybridClassifier

speakeasyConfigFile = r"C:\Users\dtrizna\Code\nebula\emulation\_speakeasyConfig.json"
vocabFile = r"C:\Users\dtrizna\Code\nebula\data\data_filtered\speakeasy_trainset_WithAPIargs\speakeasy_VocabSize_10000.pkl"
outEmulationReportFolder = r".\tempSpeakEasyReports"

model = PEHybridClassifier(
    vocabFile=vocabFile,
    speakeasyConfig=speakeasyConfigFile,
    outputFolder=outEmulationReportFolder,
)
pe = r"C:\windows\syswow64\xcopy.exe"
staticFeatures, dynamicFeatures = model.preprocess(pe)
logits = model(staticFeatures, dynamicFeatures)
print(logits)
import pdb;pdb.set_trace()