from nebula.misc import flattenList
from pefile import PEFormatError
from unicorn import UcError
from speakeasy import errors

AUDITD_FIELDS = [
    'TimeStamp',
    'rule.sidid',
    'hostname',
    'auditd.data.syscall',
    'auditd.summary.actor.primary',
    'auditd.summary.actor.secondary',
    'auditd.summary.object.primary',
    'auditd.summary.object.secondary',
    'process.title',
    'process.args',
    'process.working_directory',
    'process.ppid',
    'process.parent.process.executable',
    'process.parent.process.title'
    ]

AUDITD_TYPES = [
    'execution',
    'mount',
    'external-access',
    'log-modification',
    'user_start',
    'user_end',
    'bind-access',
    'time-change',
    'perm_modification',
    'inbound-access',
    'unsucces_file_access'
    ]

JSON_CLEANUP_SYMBOLS = ['"', "'", ":", ",", "[", "]", "{", "}", "\\", "/"]

# this defines an order in which events occur in final sequence
SPEAKEASY_RECORDS = ["registry_access", "file_access", "network_events.traffic", "apis"]

SPEAKEASY_RECORD_LIMITS = {"network_events.traffic": 256}

SPEAKEASY_RECORD_SUBFILTER = {'apis': ['api_name', 'args', 'ret_val'],
                       'file_access': ['event', 'path', 'open_flags', 'access_flags', 'size'],
                       'network_events.traffic': ['server', 'proto', 'port', 'method']}

SPEAKEASY_RECORD_SUBFILTER_MINIMALISTIC = {'apis': ['api_name', 'ret_val'],
                                    'file_access': ['event', 'path'],
                                    'network_events.traffic': ['server', 'port']}

SPEAKEASY_TOKEN_STOPWORDS = flattenList([SPEAKEASY_RECORD_SUBFILTER[x] for x in SPEAKEASY_RECORD_SUBFILTER])

SPEAKEASY_CONFIG = r"C:\Users\dtrizna\Code\nebula\emulation\_speakeasyConfig.json"

# TOP 100 RETURN VALUES, OTHERS REPLACE WITH DEDICATED PLACEHOLDER
RETURN_VALUES_TOKEEP = ['0x1', '0x0', '0xfeee0001', '0x46f0', '0x77000000', '0x4690', '0x90', '0x100', '0xfeee0004', '0x6', '0x10c', '-0x1', '0xfeee0002', '0xfeee0000', '0x54', '0x3', '0x10', '0xfeee0005', '0x2', '0xfeee0003', '0x7d90', '0xfeee0006', '0x4610', '0x45f0', '0x20', '0xffffffff', '0x4e4', '0x8810', '0x7e70', '0x7', '0x7000', '0xc000', '0xfeee0007', '0xcd', '0xf002', '0xf001', '0xf003', '0xfeee0008', '0xfeee0009', '0xfeee000b', '0xfeee000a', '0xfeee000c', '0xfeee0014', '0x47b0', '0xfeee000e', '0xfeee000d', '0xfeee000f', '0xfeee0015', '0xfeee0016', '0xfeee0010', '0xfeee0011', '0xfeee0013', '0xfeee0012', '0x4', '0xfeee0017', '0xfeee0018', '0xfeee0019', '0x8000', '0x7ec0', '0x400000', '0x1db10106', '0xfeee001a', '0xfeee001c', '0xfeee001b', '0x102', '0x5', '0xfeee0071', '0x8', '0x5265c14', '0x9000', '0x7de0', '0xc', '0x14', '0xfeee001d', '0x46d0', '0xfeee001e', '0xfeee001f', '0xfeee0020', '0x50000', '0xe', '0x8cc0', '0x4012ac', '0x12', '0xfeee0040', '0xfeee0022', '0xfeee0021', '0xfeee0023', '0xfeee0024', '0xfeee0025', '0x77d10000', '0xfeee0027', '0x2a', '0xfeee0026', '0x2c', '0xfeee007e', '0xfeee005d', '0xfeee0028', '0x78000000', '0x2e', '0xfeee007c']

# good reference:
# https://docs.microsoft.com/en-us/windows/deployment/usmt/usmt-recognized-environment-variables
VARIABLE_MAP = {
    r"%systemdrive%": r"<drive>", 
    r"%systemroot%": r"<drive>\windows",
    r"%windir%": r"<drive>\windows", 
    r"%allusersprofile%": r"<drive>\programdata",
    r"%programdata%": r"<drive>\programdata",
    r"%programfiles%": r"<drive>\program files",
    r"%programfiles(x86)%": r"<drive>\program files (x86)",
    r"%programw6432%": r"<drive>\program files",
    r"%commonprogramfiles%": r"<drive>\program files\common files",
    r"%commonprogramfiles(x86)%": r"<drive>\program files (x86)\common files",
    r"%commonprogramw6432%": r"<drive>\program files\common files",
    r"%commonfiles%": r"<drive>\program files\common files",
    r"%profiles%": r"<drive>\users",
    r"%public%": r"<drive>\users\public",
    r"%userprofile%": r"<drive>\users\<user>"
}
# more user variables
VARIABLE_MAP.update({
    r"%homepath%": VARIABLE_MAP[r"%userprofile%"],
    r"%downloads%": VARIABLE_MAP[r"%userprofile%"] + r"\downloads",
    r"%desktop%": VARIABLE_MAP[r"%userprofile%"] + r"\desktop",
    r"%favorites%": VARIABLE_MAP[r"%userprofile%"] + r"\favorites",
    r"%documents%": VARIABLE_MAP[r"%userprofile%"] + r"\documents",
    r"%mydocuments%": VARIABLE_MAP[r"%userprofile%"] + r"\documents", # obsolete
    r"%personal%": VARIABLE_MAP[r"%userprofile%"] + r"\documents", # obsolete
    r"%localsettings%": VARIABLE_MAP[r"%userprofile%"] + r"\documents", # obsolete
    r"%mypictures%": VARIABLE_MAP[r"%userprofile%"] + r"\documents\my pictures",
    r"%mymusic%": VARIABLE_MAP[r"%userprofile%"] + r"\documents\my music",
    r"%myvideos%": VARIABLE_MAP[r"%userprofile%"] + r"\documents\my videos",
    r"%localappdata%": VARIABLE_MAP[r"%userprofile%"] + r"\appdata\local",
    r"%appdata%": VARIABLE_MAP[r"%userprofile%"] + r"\appdata\roaming",
    r"%usertemp%": VARIABLE_MAP[r"%userprofile%"] + r"\appdata\local\temp",
    r"%temp%": VARIABLE_MAP[r"%userprofile%"] + r"\appdata\local\temp",
    r"%tmp%": VARIABLE_MAP[r"%userprofile%"] + r"\appdata\local\temp",
    r"%cache%": VARIABLE_MAP[r"%userprofile%"] + r"\appdata\local\microsoft\windows\temporary internet files"
})    

SPEAKEASY_EXCEPTIONS = (PEFormatError, UcError, IndexError, errors.NotSupportedError, errors.SpeakeasyError)
