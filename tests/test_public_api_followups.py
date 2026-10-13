"""Check every native follow-up arm against independent numerical references."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import public_api_followups as followups


class FollowupTests(unittest.TestCase):
    def test_all_supported_arms_and_fixture_values(self):
        import jax
        import numpy as np
        jax.config.update("jax_enable_x64",True)
        for framework in ("pytorch","jax"):
            rows=followups.cases(framework,shapes=((7,9),))
            self.assertEqual(len(rows),13)
            for row in rows:
                if row[-1] is None:
                    self.assertEqual((framework,row[1]),("pytorch","masked_softmax"))
                    continue
                row[-1].prepare()
                row[-1].validate()
        x,mask,indices=followups.fixture(7,9)
        for i,j in ((0,0),(3,4),(6,8)):
            p=i+7*j
            raw=((p*6364136223846793005+1442695040888963407)%(1<<64))%2048
            expected=np.float32(.75*(raw-1024)/1024+((i*13+j*7)%31)/32-.5)
            self.assertEqual(x[i,j],expected)
            self.assertEqual(mask[i,j],p%3==0)
            self.assertEqual(indices[i,j],(p*37+11)%9)
        self.assertGreater(float(np.max(np.abs(x.mean(axis=1)))),0.01)
        self.assertFalse(np.allclose(followups.reference("rms_norm",x,mask,indices),followups.reference("layer_norm",x,mask,indices)))


if __name__ == "__main__":unittest.main()
