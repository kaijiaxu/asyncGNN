from typing import Any, Callable, List, Optional, Tuple
from clrs._src.processors import Processor

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

_Array = chex.Array
_Fn = Callable[..., Any]
BIG_NUMBER = 1e6

class GATL1AsyncTest(Processor):
    """Graph Attention Network (Velickovic et al., ICLR 2018)."""

    def __init__(
            self,
            out_size: int,
            nb_heads: int,
            activation: Optional[_Fn] = jax.nn.relu,
            residual: bool = True,
            use_ln: bool = False,
            name: str = 'gat_l1_async_aggr',
    ):
        super().__init__(name=name)
        self.out_size = out_size
        self.nb_heads = nb_heads
        if out_size % nb_heads != 0:
            raise ValueError('The number of attention heads must divide the width!')
        self.head_size = out_size // nb_heads
        self.activation = activation
        self.residual = residual
        self.use_ln = use_ln

    def __call__(  # pytype: disable=signature-mismatch  # numpy-scalars
            self,
            node_fts: _Array,
            edge_fts: _Array,
            graph_fts: _Array,
            adj_mat: _Array,
            hidden: _Array,
            **unused_kwargs,
    ) -> _Array:
        """GAT inference step."""

        b, n, _ = node_fts.shape
        assert edge_fts.shape[:-1] == (b, n, n)
        assert graph_fts.shape[:-1] == (b,)
        assert adj_mat.shape == (b, n, n)

        z = jnp.concatenate([node_fts, hidden], axis=-1)
        m = hk.Linear(self.out_size)
        skip = hk.Linear(self.out_size)

        bias_mat = (adj_mat - 1.0) * 1e9
        bias_mat = jnp.tile(bias_mat[..., None],
                            (1, 1, 1, self.nb_heads))  # [B, N, N, H]
        bias_mat = jnp.transpose(bias_mat, (0, 3, 1, 2))  # [B, H, N, N]

        a_1 = hk.Linear(self.nb_heads)
        a_2 = hk.Linear(self.nb_heads)
        a_e = hk.Linear(self.nb_heads)
        a_g = hk.Linear(self.nb_heads)

        values = m(z)  # [B, N, H*F]
        values = jnp.reshape(
            values,
            values.shape[:-1] + (self.nb_heads, self.head_size))  # [B, N, H, F]
        values = jnp.transpose(values, (0, 2, 1, 3))  # [B, H, N, F]

        att_1 = jnp.expand_dims(a_1(z), axis=-1)
        att_2 = jnp.expand_dims(a_2(z), axis=-1)
        att_e = a_e(edge_fts)
        att_g = jnp.expand_dims(a_g(graph_fts), axis=-1)

        logits = (
                jnp.transpose(att_1, (0, 2, 1, 3)) +  # + [B, H, N, 1]
                jnp.transpose(att_2, (0, 2, 3, 1)) +  # + [B, H, 1, N]
                jnp.transpose(att_e, (0, 3, 1, 2)) +  # + [B, H, N, N]
                jnp.expand_dims(att_g, axis=-1)  # + [B, H, 1, 1]
        )  # = [B, H, N, N]

        coef = jax.nn.leaky_relu(logits) + bias_mat
        exp_coefs = jnp.exp(coef)
        res = jnp.zeros_like(values)  # same shape as values
        sigma = jnp.zeros_like(values)  # same shape as values
        index = np.arange(0, n*self.head_size*n, 1)
        np.random.shuffle(index)
        for num in index:
            k = num // (n*n)
            rem = num % (n*n)
            j = rem // n
            i = rem % n
            numerator = jnp.multiply(res[:, :, i, k], sigma[:, :, i, k]) + jnp.multiply(exp_coefs[:, :, i, j], values[:, :, j, k])
            denominator = sigma[:, :, i, k] + exp_coefs[:, :, i, j]
            new_val = jnp.where(denominator == 0, 0, numerator/denominator)  # deal with division by zero
            res = res.at[:, :, i, k].set(new_val)
            sigma = sigma.at[:, :, i, k].add(exp_coefs[:, :, i, j])
        ret = res

        ret = jnp.transpose(ret, (0, 2, 1, 3))  # [B, N, H, F]
        ret = jnp.reshape(ret, ret.shape[:-2] + (self.out_size,))  # [B, N, H*F]

        if self.residual:
            ret += skip(z)

        if self.activation is not None:
            ret = self.activation(ret)

        if self.use_ln:
            ln = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
            ret = ln(ret)

        return ret, None  # pytype: disable=bad-return-type  # numpy-scalars


class GATv2L1AsyncTest(Processor):
    """Graph Attention Network v2 (Brody et al., ICLR 2022)."""

    def __init__(
            self,
            out_size: int,
            nb_heads: int,
            mid_size: Optional[int] = None,
            activation: Optional[_Fn] = jax.nn.relu,
            residual: bool = True,
            use_ln: bool = False,
            name: str = 'gatv2_l1_async_aggr',
    ):
        super().__init__(name=name)
        if mid_size is None:
            self.mid_size = out_size
        else:
            self.mid_size = mid_size
        self.out_size = out_size
        self.nb_heads = nb_heads
        if out_size % nb_heads != 0:
            raise ValueError('The number of attention heads must divide the width!')
        self.head_size = out_size // nb_heads
        if self.mid_size % nb_heads != 0:
            raise ValueError('The number of attention heads must divide the message!')
        self.mid_head_size = self.mid_size // nb_heads
        self.activation = activation
        self.residual = residual
        self.use_ln = use_ln

    def __call__(  # pytype: disable=signature-mismatch  # numpy-scalars
            self,
            node_fts: _Array,
            edge_fts: _Array,
            graph_fts: _Array,
            adj_mat: _Array,
            hidden: _Array,
            **unused_kwargs,
    ) -> _Array:
        """GATv2 inference step."""

        b, n, _ = node_fts.shape
        assert edge_fts.shape[:-1] == (b, n, n)
        assert graph_fts.shape[:-1] == (b,)
        assert adj_mat.shape == (b, n, n)

        z = jnp.concatenate([node_fts, hidden], axis=-1)
        m = hk.Linear(self.out_size)
        skip = hk.Linear(self.out_size)

        bias_mat = (adj_mat - 1.0) * 1e9
        bias_mat = jnp.tile(bias_mat[..., None],
                            (1, 1, 1, self.nb_heads))  # [B, N, N, H]
        bias_mat = jnp.transpose(bias_mat, (0, 3, 1, 2))  # [B, H, N, N]

        w_1 = hk.Linear(self.mid_size)
        w_2 = hk.Linear(self.mid_size)
        w_e = hk.Linear(self.mid_size)
        w_g = hk.Linear(self.mid_size)

        a_heads = []
        for _ in range(self.nb_heads):
            a_heads.append(hk.Linear(1))

        values = m(z)  # [B, N, H*F]
        values = jnp.reshape(
            values,
            values.shape[:-1] + (self.nb_heads, self.head_size))  # [B, N, H, F]
        values = jnp.transpose(values, (0, 2, 1, 3))  # [B, H, N, F]

        pre_att_1 = w_1(z)
        pre_att_2 = w_2(z)
        pre_att_e = w_e(edge_fts)
        pre_att_g = w_g(graph_fts)

        pre_att = (
                jnp.expand_dims(pre_att_1, axis=1) +  # + [B, 1, N, H*F]
                jnp.expand_dims(pre_att_2, axis=2) +  # + [B, N, 1, H*F]
                pre_att_e +  # + [B, N, N, H*F]
                jnp.expand_dims(pre_att_g, axis=(1, 2))  # + [B, 1, 1, H*F]
        )  # = [B, N, N, H*F]

        pre_att = jnp.reshape(
            pre_att,
            pre_att.shape[:-1] + (self.nb_heads, self.mid_head_size)
        )  # [B, N, N, H, F]

        pre_att = jnp.transpose(pre_att, (0, 3, 1, 2, 4))  # [B, H, N, N, F]

        # This part is not very efficient, but we agree to keep it this way to
        # enhance readability, assuming `nb_heads` will not be large.
        logit_heads = []
        for head in range(self.nb_heads):
            logit_heads.append(
                jnp.squeeze(
                    a_heads[head](jax.nn.leaky_relu(pre_att[:, head])),
                    axis=-1)
            )  # [B, N, N]

        logits = jnp.stack(logit_heads, axis=1)  # [B, H, N, N]

        # coefs = jax.nn.softmax(logits + bias_mat, axis=-1)
        # ret = jnp.matmul(coefs, values)  # [B, H, N, F]

        coef = logits + bias_mat
        exp_coefs = jnp.exp(coef)
        res = jnp.zeros_like(values)  # same shape as values
        sigma = jnp.zeros_like(values)  # same shape as values

        index = np.arange(0, n*self.head_size*n, 1)
        np.random.shuffle(index)
        for num in index:
            k = num // (n*n)
            rem = num % (n*n)
            j = rem // n
            i = rem % n
            numerator = jnp.multiply(res[:, :, i, k], sigma[:, :, i, k]) + jnp.multiply(exp_coefs[:, :, i, j], values[:, :, j, k])
            denominator = sigma[:, :, i, k] + exp_coefs[:, :, i, j]
            new_val = jnp.where(denominator == 0, 0, numerator/denominator)  # deal with division by zero
            res = res.at[:, :, i, k].set(new_val)
            sigma = sigma.at[:, :, i, k].add(exp_coefs[:, :, i, j])
        ret = res

        ret = jnp.transpose(ret, (0, 2, 1, 3))  # [B, N, H, F]
        ret = jnp.reshape(ret, ret.shape[:-2] + (self.out_size,))  # [B, N, H*F]

        if self.residual:
            ret += skip(z)

        if self.activation is not None:
            ret = self.activation(ret)

        if self.use_ln:
            ln = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
            ret = ln(ret)

        return ret, None  # pytype: disable=bad-return-type  # numpy-scalars