
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

FIELD_SEPARATOR = " , "

SPEAKEASY_RECORDS = ["apis", "registry_access", "file_access", 'network_events.traffic']

SPEAKEASY_SUBFILTER = {'apis': ['api_name', 'args', 'ret_val'],
                       'file_access': ['event', 'path', 'open_flags', 'access_flags', 'size'],
                       'network_events.traffic': ['server', 'proto', 'port', 'method']}

SPEAKEASY_SUBFILTER_MINIMALISTIC = {'apis': ['api_name', 'ret_val'],
                                    'file_access': ['event', 'path'],
                                    'network_events.traffic': ['server', 'port']}

# good reference:
# https://docs.microsoft.com/en-us/windows/deployment/usmt/usmt-recognized-environment-variables
VARIABLE_MAP = {
    r"%systemdrive%": r"[drive]", 
    r"%systemroot%": r"[drive]\windows",
    r"%windir%": r"[drive]\windows", 
    r"%allusersprofile%": r"[drive]\programdata",
    r"%programdata%": r"[drive]\programdata",
    r"%programfiles%": r"[drive]\program files",
    r"%programfiles(x86)%": r"[drive]\program files (x86)",
    r"%programw6432%": r"[drive]\program files",
    r"%commonprogramfiles%": r"[drive]\program files\common files",
    r"%commonprogramfiles(x86)%": r"[drive]\program files (x86)\common files",
    r"%commonprogramw6432%": r"[drive]\program files\common files",
    r"%commonfiles%": r"[drive]\program files\common files",
    r"%profiles%": r"[drive]\users",
    r"%public%": r"[drive]\users\public",
    r"%userprofile%": r"[drive]\users\[user]"
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

