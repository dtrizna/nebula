import os
import io
import yara
import py7zr
import zipfile
import requests
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def get_encrypted_archive(
        link: str = None,
        user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
        password: str = 'infected',
        remove_archive: bool = None
) -> bytes:
    if link.startswith("http://") or link.startswith("https://"):
        archive_name = link.split("/")[-1]
        with requests.get(link, headers={"User-Agent": user_agent}) as response:
            response.raise_for_status()
            archive = response.content
        
        with open(archive_name, "wb") as f:
            f.write(archive)
        if remove_archive is None:
            remove_archive = True
    else:
        archive_name = link
        if remove_archive is None:
            remove_archive = False
    
    if archive_name.endswith(".7z"):
        # vx-underground, single file <hash>.7z with <hash> inside
        file_hash = os.path.basename(archive_name).replace(".7z", "")
        with py7zr.SevenZipFile(archive_name, "r", password=password) as archive:
            try:
                content = archive.read(targets=file_hash)[file_hash].read()
            except KeyError: # NOTE: not tested
                print(f"[-] File {file_hash} not found in archive {archive_name}, providing all files")
                content = {file: archive.read(file) for file in archive.getnames()}

    elif archive_name.endswith(".zip"):
        # other sources, multiple files
        with zipfile.ZipFile(archive_name, "r") as archive:
            archive.setpassword(password.encode())
            content = {file: archive.read(file) for file in archive.namelist()}
    else:
        raise ValueError(f"[-] archive must be .7z or .zip, got: {archive_name}")

    if remove_archive:
        os.remove(archive_name)

    return content

class MalConv(nn.Module):
    # trained to minimize cross-entropy loss
    # criterion = nn.CrossEntropyLoss()
    def __init__(self, out_size=2, channels=128, window_size=512, embd_size=8):
        super(MalConv, self).__init__()
        self.embd = nn.Embedding(257, embd_size, padding_idx=0)
        
        self.window_size = window_size
    
        self.conv_1 = nn.Conv1d(embd_size, channels, window_size, stride=window_size, bias=True)
        self.conv_2 = nn.Conv1d(embd_size, channels, window_size, stride=window_size, bias=True)
        
        self.pooling = nn.AdaptiveMaxPool1d(1)
        
        self.fc_1 = nn.Linear(channels, channels)
        self.fc_2 = nn.Linear(channels, out_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embd(x.long())
        x = torch.transpose(x,-1,-2)
        
        cnn_value = self.conv_1(x)
        gating_weight = torch.sigmoid(self.conv_2(x))
        
        x = cnn_value * gating_weight
        
        x = self.pooling(x)
        
        # flatten
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc_1(x))
        x = self.fc_2(x)
        
        return x


class MalConvModel(object):
    def __init__(self): 
        self.model = MalConv(channels=256, window_size=512, embd_size=8)#.train()
        
    def load_state(self, model_weights):
        if isinstance(model_weights, str): # on disk
            weights = torch.load(model_weights, map_location='cpu')
        elif isinstance(model_weights, bytes): # memory
            weights = torch.load(io.BytesIO(model_weights), map_location='cpu')
        elif isinstance(model_weights, dict): # dict
            weights = model_weights
        else:
            raise ValueError("model_weights must be a file path, bytes, or dict")
        self.model.load_state_dict(weights['model_state_dict'])

    def get_score_from_bytez(self, bytez):
        bytez = bytez[:2000000]
        try:
            _inp = torch.from_numpy( np.frombuffer(bytez, dtype=np.uint8)[np.newaxis,:].copy() )
            with torch.no_grad():
                outputs = F.softmax( self.model(_inp), dim=-1)
            return outputs.detach().numpy()[0,1]
        except Exception as e:
            print(e)
        return 0.0

    def get_score_from_path(self, file_path):
        try:
            with open(file_path, 'rb') as fp:
                bytez = fp.read(2000000) # read the first 2000000 bytes
            return self.get_score_from_bytez(bytez)
        except Exception as e:
            print(e)
        return 0.0
    
    def get_score(self, input_data):
        if isinstance(input_data, str):
            return self.get_score_from_path(input_data)
        elif isinstance(input_data, bytes):
            return self.get_score_from_bytez(input_data)
        else:
            raise ValueError("input_data must be a file path or bytes")

    def is_evasive(self, input_data, threshold = 0.5):
        score = self.get_score(input_data)
        return score < threshold

class YaraWrapper(object):
    def __init__(self) -> None:
        pass
        
    def check_sample(self, sample_bytes, rules) -> yara.Match:
        rules = yara.compile(source=rules)
        self.matches = rules.match(data=sample_bytes)
        if self.matches:
            print(f"[+] Match found: {self.matches}")
        else:
            print("[-] No matches found.")
        return self.matches

    def pretty_print(self, print_limit=10):
        for match in self.matches:
            print(f"[!] Rule: {match.rule}")
            for i, string in enumerate(match.strings):
                for j, instance in enumerate(string.instances):
                    if j >= print_limit:
                        print(f"    [*] ... {len(string.instances) - print_limit} more\n")
                        break

                    print(f"    [+] Matched data: {instance.matched_data} | Offset: 0x{instance.offset:x}")
                    if string.is_xor():
                        print(f"\t[!] XOR key: {instance.xor_key}")
                        print(f"\t[!] Decoded: {instance.plaintext()}")
                
                if i >= print_limit:
                    print(f"\n  [*] ... {len(match.strings) - print_limit} more")
                    break
        