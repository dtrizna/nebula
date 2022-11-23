import re
import subprocess
import time

def normalize_ip(x):
    return re.sub(r"(?:[0-9]{1,3}\.){3}[0-9]{1,3}", "127.0.0.1", x)


def normalize_dataset():
    with open("../data/reverse_shell_dataset.txt") as f:
        dataset_raw = f.readlines()

    dataset_normalized = [normalize_ip(x) for x in dataset_raw]
    
    with open("../data/reverse_shell_dataset_normalized.txt", "w") as f:
        f.writelines(dataset_normalized)


if __name__ == "__main__":

    normalize_dataset()

    with open("../data/reverse_shell_dataset_normalized.txt") as f:
        dataset_normalized = f.readlines()

    # # execute
    for cmd in dataset_normalized[1000:]:
        print(cmd)
        try:
            out = subprocess.check_output(cmd.strip(), shell=True, stderr=subprocess.STDOUT, timeout=1)
        except (subprocess.CalledProcessError, FileNotFoundError) as ex:
            print(f"Got exception: {ex}")
            pass
        import pdb;pdb.set_trace()
        time.sleep(0.01)
