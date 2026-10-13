"""Public JAX Ref updates with scalar device-completion tokens.

Full ref.get snapshots are reserved for untimed numerical inspection.
"""


def prepare_addupdate(jax, case, inputs, *, compiled, verify_only=False):
    if case.get("alpha") != 1. or case.get("beta") != 1.:
        raise ValueError("Ref.addupdate has no scaled alpha/beta arguments")
    scalar_bytes = inputs[0].dtype.itemsize
    cap = 1 if verify_only else max(1, min(4096, (256 << 20) // (inputs[0].nbytes + scalar_bytes)))
    destinations = [jax.new_ref(inputs[1]) for _ in range(cap)]
    pointers = {id(ref): ref.unsafe_buffer_pointer() for ref in destinations}
    if len(set(pointers.values())) != cap:
        raise ValueError("verification Ref destinations alias")
    waits = {}

    def update(ref, value):
        jax.ref.addupdate(ref, ..., value)

    if compiled:
        update = jax.jit(update).lower(destinations[0], inputs[0]).compile()

    def scalar_tokens(refs):
        return tuple(jax.ref.get(ref, 0) for ref in refs)

    def ready(outputs):
        tokens = waits[len(outputs)](tuple(outputs))
        jax.block_until_ready(tokens)
        return tokens  # caller retains synchronization outputs through clock stop

    def prepare(count):
        refs = tuple(destinations[:count])
        if count not in waits:
            waits[count] = jax.jit(scalar_tokens).lower(refs).compile()
        for ref in refs:
            jax.ref.set(ref, ..., inputs[1])
        ready(refs)  # restore and initialize synchronization outside timing

    def operation(index):
        ref = destinations[index]
        update(ref, inputs[0])
        return ref

    def inspect(ref):
        if ref.unsafe_buffer_pointer() != pointers[id(ref)]:
            raise ValueError("verification Ref destination buffer changed")
        return jax.ref.get(ref, ...)

    operation.prepare = prepare
    operation.ready = ready
    operation.inspect = inspect
    operation.synchronization_bytes_per_operation = scalar_bytes
    return operation
