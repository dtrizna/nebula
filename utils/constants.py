
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
