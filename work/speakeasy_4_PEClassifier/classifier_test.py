import sys
sys.path.extend(['../..', '.'])
from nebula import PEHybridClassifier

speakeasyConfigFile = r"C:\Users\dtrizna\Code\nebula\emulation\_speakeasyConfig.json"
vocabFile = r"C:\Users\dtrizna\Code\nebula\data\data_filtered\speakeasy_trainset_WithAPIargs\speakeasy_VocabSize_10000.pkl"
outEmulationReportFolder = r"C:\Users\dtrizna\Code\nebula\tests\speakeasy_4_PEClassifier\tempSpeakEasyReports"

model = PEHybridClassifier(
    vocabFile=vocabFile,
    speakeasyConfig=speakeasyConfigFile,
    outputFolder=outEmulationReportFolder,
)
testFile = r"C:\windows\syswow64\xcopy.exe"
static, dynamic = model.preprocess(testFile)

logits = model(static, dynamic)
print(logits)
import pdb;pdb.set_trace()