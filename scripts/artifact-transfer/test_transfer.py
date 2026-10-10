import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from diagnose import analyze
from experiment import CASES, begin, end, evaluate, prepare

class TransferTests(unittest.TestCase):
    def test_interleaved_downloads_are_joined_by_digest_not_completion_order(self):
        a,b='a'*64,'b'*64
        log='\n'.join('job\tstep\t2026-10-10T16:00:'+s for s in [
            f'00Z - first (ID: 1, Size: 20, Expected Digest: sha256:{a})',
            f'00Z - second (ID: 2, Size: 30, Expected Digest: sha256:{b})',
            "01Z Downloading artifact '1'", "02Z Downloading artifact '2'",
            f'04Z SHA256 digest of downloaded artifact is {b}',
            f'09Z SHA256 digest of downloaded artifact is {a}'])
        rows=analyze(log)[0]['artifacts']
        self.assertEqual([r['seconds_to_digest'] for r in rows],[8,2])
        self.assertNotIn('seconds_to_digest',analyze(log.rsplit('\n',1)[0])[0]['artifacts'][0])

    def test_real_split_payload_roundtrip_and_corruption_rejection(self):
        with tempfile.TemporaryDirectory() as temp:
            old=os.getcwd(); os.chdir(temp)
            try:
                for case,stem in CASES.items():
                    p=Path('source')/case; p.mkdir(parents=True)
                    pieces=[bytes([i])*19 for i in range(5)]
                    for i,data in enumerate(pieces): (p/f'{stem}.part{i:02}').write_bytes(data)
                    (p/f'{stem}.sha256').write_text(hashlib.sha256(b''.join(pieces)).hexdigest()+f'  {stem}.tar.zst\n')
                prepare()
                for arm in ['A','B']:
                    begin('0','common')
                    import shutil
                    for i in range(5):
                        src=Path(f'upload/zip/common/{i:02}') if arm=='A' else Path('upload/raw')
                        for path in src.glob('*' if arm=='A' else f'raw-common-part{i:02}'):
                            shutil.copyfile(path,Path('received')/path.name)
                    end(arm)
                (Path('received')/'raw-common-part02').write_bytes(b'bad')
                with self.assertRaises(AssertionError): end('B')
            finally: os.chdir(old)

    def test_complete_gate_rejects_missing_noisy_regressed_or_bad_bytes(self):
        protocol={'order':['A','A','B','A','B','A','A','B','A','B'],'aa_max_min':1.15,'min_reduction':.2,'per_case_max_median_ratio':1.1}
        rows=[{'index':i,'case':c,'arm':a,'seconds':10 if a=='A' else 6,'verified':True}
              for i,a in enumerate(protocol['order']) for c in CASES]
        self.assertEqual(evaluate(rows,protocol)['verdict'],'PASS')
        self.assertEqual(evaluate(rows[:-1],protocol)['verdict'],'INCONCLUSIVE')
        for key,value,verdict in [('seconds',20,'INCONCLUSIVE'),('verified',False,'INCONCLUSIVE')]:
            changed=copy.deepcopy(rows);changed[0][key]=value
            self.assertEqual(evaluate(changed,protocol)['verdict'],verdict)
        changed=copy.deepcopy(rows)
        for r in changed:
            if r['arm']=='B':r['seconds']=12
        self.assertEqual(evaluate(changed,protocol)['verdict'],'FAIL')

if __name__=='__main__':unittest.main()
