"""Inspect actual ATen GEMM operands; not a timing benchmark."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / 'scripts'))
import benchmark_python as bench
bench.configure_thread_env(1)
torch = bench.configure_pytorch_threads(1)
import opt_einsum as oe
from torch.utils._python_dispatch import TorchDispatchMode

class Shapes(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func in (torch.ops.aten.bmm.default, torch.ops.aten.mm.default):
            print(json.dumps({'op': str(func), 'inputs': [{'shape': list(t.shape), 'stride': list(t.stride())} for t in args[:2]]}))
        return func(*args, **(kwargs or {}))

instance, = bench.load_instances('lm_batch_likelihood_sentence_3_12d')
operands = [torch.zeros(s, dtype=torch.float64) for s in instance['shapes']]
for strategy in ('opt_flops', 'opt_size'):
    print(json.dumps({'strategy': strategy, 'threads': torch.get_num_threads()}))
    expression = oe.contract_expression(bench.get_format_string(instance), *instance['shapes'], optimize=bench.path_to_opt_einsum(instance['paths'][strategy]['path']))
    with Shapes():
        expression(*operands, backend='torch')
