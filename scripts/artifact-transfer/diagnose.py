"""Map existing official download logs to per-artifact wall times.

The interval starts at the ID-specific download request and ends at that
artifact's digest log. It includes HTTP setup, network, unzip and hashing;
these existing logs cannot isolate first-byte latency or unzip CPU time.
The pinned action hides retry details behind core.debug; an empty message
list does not establish that no retries occurred.
"""
import datetime as dt
import json
from pathlib import Path
import re
import sys


def analyze(log):
    groups = {}
    for line in log.splitlines():
        fields = line.split('\t', 2)
        if len(fields) != 3: continue
        job, step, message = fields
        match = re.match(r'\ufeff?(\d{4}-\d\d-\d\dT\S+Z) (.*)', message)
        if not match: continue
        timestamp = dt.datetime.fromisoformat(match[1].replace('Z','+00:00'))
        message = match[2]
        group = groups.setdefault((job,step), {'artifacts': {}, 'digest_ids': {}, 'retries': []})
        item = re.search(r'- (.+) \(ID: (\d+), Size: (\d+), Expected Digest: sha256:([a-f0-9]{64})\)', message)
        if item:
            name, ident, size, digest = item.groups()
            group['artifacts'][ident] = {'name': name, 'id': int(ident), 'bytes': int(size), 'digest': digest}
            group['digest_ids'].setdefault(digest, []).append(ident)
        start = re.search(r"Downloading artifact '(\d+)'", message)
        if start and start[1] in group['artifacts']:
            group['artifacts'][start[1]]['started_at'] = timestamp.isoformat()
        end = re.search(r'SHA256 digest of downloaded artifact is ([a-f0-9]{64})', message)
        if end:
            ids = group['digest_ids'].get(end[1], [])
            if len(ids)==1:
                row = group['artifacts'][ids[0]]
                row['digest_at'] = timestamp.isoformat()
                if 'started_at' in row:
                    row['seconds_to_digest'] = (timestamp-dt.datetime.fromisoformat(row['started_at'])).total_seconds()
        if re.search(r'retry|retrying|timed out|timeout|ECONN', message, re.I) and not '\x1b' in message and not '^[' in message:
            group['retries'].append(message)
    return [{'job':j, 'step':s, 'artifacts':list(g['artifacts'].values()), 'retry_messages':g['retries'],
             'retry_visibility':'debug-only; absence of messages does not imply no retries'}
            for (j,s),g in groups.items() if g['artifacts']]


if __name__ == '__main__':
    print(json.dumps(analyze(Path(sys.argv[1]).read_text()), indent=2))
