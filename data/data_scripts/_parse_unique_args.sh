#!/bin/bash

echo [*] Getting unique commands from raw folder...

for i in $(ls auditd_msft_raw); do
    grep process.args auditd_raw/$i | sort -u | cut -d\" -f4 >> args.tmp
done

echo [*] Normalizing commands...

sort -u args.tmp > auditd_msft_ArgsNormalizedUnique.cm
rm args.tmp

LINES=$(wc -l auditd_msft_ArgsNormalizedUnique.cm | awk '{print $1}')
echo You have $LINES unique entries from MSFT auditd!