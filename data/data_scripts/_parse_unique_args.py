import re, os, time

AUDITD_FOLDER = "auditd_msft_raw"
OUTPUT_FILE = "auditd_msft_ArgsNormalizedUnique.cm"

def normalizeCommand(cmd):
    cmd = re.sub(r"([0-9a-f]{64}|[0-9a-f]{32}|[0-9a-f]{26})", "_HASH_", cmd)
    cmd = re.sub(r"(?:[0-9]{1,3}\.){3}[0-9]{1,3}", "_IPADDRESS_", cmd)
    cmd = re.sub(r"[0-9]{3,8}", "_NUMBER_", cmd)
    #cmd = re.sub(r"\.?[a-z0-9A-Z]{10}\.?", "_RANDOMID_", cmd)
    return cmd if cmd.endswith("\n") else cmd+"\n"

if __name__ == "__main__":
    now = time.time()
    uniqueCommands = set()

    files = os.listdir(AUDITD_FOLDER)
    l = len(files)
    for i in range(l):
        print(f"[*] Parsing files in ./{AUDITD_FOLDER}/: {i}/{l}", end="\r")
        with open(os.path.join(AUDITD_FOLDER, files[i]), "r") as f:
            data = f.readlines()
        args = [x for x in data if "process.args" in x]
        args = [x.split('"')[3] for x in args]
        uniqueCommands.update(set(args))

    normalizedCommands = set()
    print(f"[*] Normalizing commands...")
    normalizedCommands = set([normalizeCommand(x) for x in uniqueCommands])
    with open(OUTPUT_FILE, "w") as f:
        f.writelines(normalizedCommands)

    print(len(uniqueCommands))
    print(f"[!] Written {len(normalizedCommands)} unique commands... Took: {time.time()-now:.2f}s")