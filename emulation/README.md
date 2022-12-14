# Reverse Shell Simulation

Failed...

With this call:

```python
out = subprocess.check_output(cmd.strip(), shell=True, stderr=subprocess.STDOUT, timeout=1)
```

... auditbeat logs the same command as in dataset, but prefixed with `/bin/sh -c`.

With this call (no `shell=True`):

```python
out = subprocess.check_output(cmd.split(), stderr=subprocess.STDOUT, timeout=1)
```

... many commands fail before appearing in auditbeat, since generated shells are not present on system.
