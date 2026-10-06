"""Remove per-step memory copies in HF T5 attention during cached generation (math unchanged).

In transformers 4.46 the cached key/value states are stored as non-contiguous transposed views and the
score is computed as matmul(Q, K^T); torch.matmul then clones the full (N, heads, src, d) key and value
tensors at every decoding step and layer. We (1) store the states contiguously and (2) compute the
score as (K Q^T)^T, which is the same product but only needs the tiny query to be copied.
"""
import inspect, textwrap
import transformers.models.t5.modeling_t5 as m

def enable():
    if getattr(m.T5Attention, '_fast_patched', False):
        return
    src = textwrap.dedent(inspect.getsource(m.T5Attention.forward))
    reps = [
        ("key_states = key_states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)",
         "key_states = key_states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2).contiguous()"),
        ("value_states = value_states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)",
         "value_states = value_states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2).contiguous()"),
        ("scores = torch.matmul(query_states, key_states.transpose(3, 2))",
         "scores = torch.matmul(key_states, query_states.transpose(3, 2)).transpose(3, 2)"),
    ]
    for old, new in reps:
        if src.count(old) != 1:
            # different transformers version: keep the stock attention (slower, same results)
            import warnings
            warnings.warn('fast_t5: T5Attention source not recognised; using the default implementation')
            return
        src = src.replace(old, new)
    ns = {}
    exec(compile(src, '<fast_t5_attention>', 'exec'), m.__dict__, ns)
    m.T5Attention.forward = ns['forward']
    m.T5Attention._fast_patched = True
