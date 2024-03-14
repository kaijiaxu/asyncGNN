from typing import Any, Callable, List, Optional, Tuple
from clrs._src.processors import Processor, Logsemiring, Maxsemiring

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

_Array = chex.Array
_Fn = Callable[..., Any]
BIG_NUMBER = 1e6


class PGNL1AsyncTest(Processor):

    def __init__(
            self,
            out_size: int,
            mid_size: Optional[int] = None,
            mid_act: Optional[_Fn] = None,
            activation: Optional[_Fn] = jax.nn.relu,
            msgs_mlp_sizes: Optional[List[int]] = None,
            use_ln: bool = False,
            gated: bool = False,
            name: str = 'mpnn_l1_async_aggr',
    ):
        super().__init__(name=name)
        if mid_size is None:
            self.mid_size = out_size
        else:
            self.mid_size = mid_size
        self.out_size = out_size
        self.mid_act = mid_act
        self.activation = activation
        self._msgs_mlp_sizes = msgs_mlp_sizes
        self.use_ln = use_ln
        self.gated = gated

    def __call__(  # pytype: disable=signature-mismatch  # numpy-scalars
            self,
            node_fts: _Array,
            edge_fts: _Array,
            graph_fts: _Array,
            adj_mat: _Array,
            hidden: _Array,
            **unused_kwargs,
    ) -> _Array:
        """MPNN inference step.
    Level 1 of asynchronous GNNs:
      a GNN with a sum aggregator,
      a linear layer with ReLU activation for the update function,
      and an MLP for the message function.
    Asynchronous message aggregation implementation """

        b, n, _ = node_fts.shape
        assert edge_fts.shape[:-1] == (b, n, n)
        assert graph_fts.shape[:-1] == (b,)
        assert adj_mat.shape == (b, n, n)

        z = jnp.concatenate([node_fts, hidden], axis=-1)
        m_1 = hk.Linear(self.mid_size)
        m_2 = hk.Linear(self.mid_size)
        m_e = hk.Linear(self.mid_size)
        m_g = hk.Linear(self.mid_size)

        o1 = hk.Linear(self.out_size)
        o2 = hk.Linear(self.out_size)

        msg_1 = m_1(z)
        msg_2 = m_2(z)
        msg_e = m_e(edge_fts)
        msg_g = m_g(graph_fts)

        msgs = (
                jnp.expand_dims(msg_1, axis=1) + jnp.expand_dims(msg_2, axis=2) +
                msg_e + jnp.expand_dims(msg_g, axis=(1, 2)))

        if self._msgs_mlp_sizes is not None:
            msgs = hk.nets.MLP(self._msgs_mlp_sizes)(jax.nn.relu(msgs))

        if self.mid_act is not None:
            msgs = self.mid_act(msgs)

        indices = np.arange(0, n, 1)
        np.random.shuffle(indices)  # randomize sequence
        expanded_adj_mat = jnp.expand_dims(adj_mat, -1)
        new_msgs = jnp.zeros(node_fts.shape)
        i = 0
        while i < n:
            msgs_to_send = np.random.randint(0, n - i + 1)
            for _ in range(msgs_to_send):
                index = indices[i]
                agg_val = jnp.sum(msgs[:, :, index:index + 1, :] * expanded_adj_mat[:, :, index:index + 1, :], axis=1)
                new_msgs = new_msgs.at[:, index:index + 1, :].set(agg_val)
                i += 1

        msgs = new_msgs

        h_1 = o1(z)
        h_2 = o2(msgs)

        ret = h_1 + h_2

        if self.activation is not None:
            ret = self.activation(ret)

        if self.use_ln:
            ln = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
            ret = ln(ret)

        if self.gated:
            gate1 = hk.Linear(self.out_size)
            gate2 = hk.Linear(self.out_size)
            gate3 = hk.Linear(self.out_size, b_init=hk.initializers.Constant(-3))
            gate = jax.nn.sigmoid(gate3(jax.nn.relu(gate1(z) + gate2(msgs))))
            ret = ret * gate + hidden * (1 - gate)

        return ret, None  # pytype: disable=bad-return-type  # numpy-scalars


class PGNL2AsyncTest(Processor):
    """Pointer Graph Networks (Veličković et al., NeurIPS 2020)."""

    def __init__(
            self,
            out_size: int,
            activation: Optional[_Fn] = None,
            use_ln: bool = False,
            gated: bool = False,
            name: str = 'mpnn_l2_aggr',
    ):
        super().__init__(name=name)
        self.out_size = out_size
        self.activation = activation
        self.use_ln = use_ln
        self.gated = gated

    def __call__(  # pytype: disable=signature-mismatch  # numpy-scalars
            self,
            node_fts: _Array,
            edge_fts: _Array,
            graph_fts: _Array,
            adj_mat: _Array,
            hidden: _Array,
            random_msgs_to_send: _Array,
            generate_msgs: bool,
            msgs: Optional[_Array],
            **unused_kwargs,
    ) -> _Array:
        """MPNN inference step.
    Level 2 of asynchrony-invariant GNNs:
      a GNN with a max aggregator,
      update function = max,
      and a linear layer for the message function."""

        b, n, _ = node_fts.shape
        assert edge_fts.shape[:-1] == (b, n, n)
        assert graph_fts.shape[:-1] == (b,)
        assert adj_mat.shape == (b, n, n)

        if generate_msgs:
            z = jnp.concatenate([node_fts, hidden], axis=-1)
            m_1 = hk.Linear(self.out_size)
            m_2 = hk.Linear(self.out_size)
            m_e = hk.Linear(self.out_size)
            m_g = hk.Linear(self.out_size)

            msg_1 = m_1(z)
            msg_2 = m_2(z)
            msg_e = m_e(edge_fts)
            msg_g = m_g(graph_fts)

            msgs = (
                    jnp.expand_dims(msg_1, axis=1) + jnp.expand_dims(msg_2, axis=2) +
                    msg_e + jnp.expand_dims(msg_g, axis=(1, 2)))

        sent_msgs = jnp.multiply(jnp.expand_dims(adj_mat, -1), random_msgs_to_send)

        unsent_msgs = jnp.expand_dims(adj_mat, -1) - sent_msgs

        # aggregation
        maxarg_sent = jnp.where(sent_msgs, msgs, -BIG_NUMBER)
        acc_msgs_sent = jnp.max(maxarg_sent, axis=1)

        # update
        ret = jnp.maximum(node_fts, acc_msgs_sent)

        return ret, None, msgs, unsent_msgs  # pytype: disable=bad-return-type  # numpy-scalars


class MPNNL1AsyncTest(PGNL1AsyncTest):
    """Message-Passing Neural Network (Gilmer et al., ICML 2017)."""

    def __call__(self, node_fts: _Array, edge_fts: _Array, graph_fts: _Array,
                 adj_mat: _Array, hidden: _Array, **unused_kwargs) -> _Array:
        adj_mat = jnp.ones_like(adj_mat)
        return super().__call__(node_fts, edge_fts, graph_fts, adj_mat, hidden)


class MPNNL2AsyncTest(PGNL2AsyncTest):
    """Message-Passing Neural Network (Gilmer et al., ICML 2017)."""

    def __call__(self, node_fts: _Array, edge_fts: _Array, graph_fts: _Array,
                 adj_mat: _Array, hidden: _Array, random_msgs_to_send: _Array,
                 generate_msgs: bool, msgs: Optional[_Array], **unused_kwargs) -> _Array:
        adj_mat = jnp.ones_like(adj_mat)
        return super().__call__(node_fts, edge_fts, graph_fts, adj_mat, hidden,
                                random_msgs_to_send, generate_msgs, msgs)


class MPNNL3AsyncTest(Processor):
    """Pointer Graph Networks (Veličković et al., NeurIPS 2020)."""

    def __init__(
            self,
            out_size: int,
            name: str = 'mpnn_l3_aggr',
            semiring: str = 'logsumexp',
    ):
        super().__init__(name=name)
        self.out_size = out_size
        self.semiring = semiring

    def __call__(  # pytype: disable=signature-mismatch  # numpy-scalars
            self,
            node_fts: _Array,
            edge_fts: _Array,
            graph_fts: _Array,
            adj_mat: _Array,
            hidden: _Array,
            updated_nodes: _Array,
            msgs_sent: _Array,
            random_msgs_to_send: _Array,
            old_msg1: _Array,
            old_msg2: _Array,
            **unused_kwargs,
    ) -> _Array:
        """MPNN inference step.
    Level 3 of asynchrony-invariant GNNs:
      a GNN with a max aggregator,
      update function = max,
      and a max-semiring 'quadrilinear' layer for the message function."""

        b, n, _ = node_fts.shape
        assert edge_fts.shape[:-1] == (b, n, n)
        assert graph_fts.shape[:-1] == (b,)
        assert adj_mat.shape == (b, n, n)

        adj_mat = jnp.ones_like(adj_mat)

        z = jnp.concatenate([node_fts, hidden], axis=-1)

        # need to change the Linears:
        m_1 = Logsemiring(self.out_size)
        m_2 = Logsemiring(self.out_size)
        m_e = Logsemiring(self.out_size)
        m_g = Logsemiring(self.out_size)

        if self.semiring == 'maxplus':
            # need to change the Linears:
            m_1 = Maxsemiring(self.out_size)
            m_2 = Maxsemiring(self.out_size)
            m_e = Maxsemiring(self.out_size)
            m_g = Maxsemiring(self.out_size)

        msg_1 = m_1(z)
        msg_2 = m_2(z)
        msg_e = m_e(edge_fts)
        msg_g = m_g(graph_fts)

        old_updated_nodes = updated_nodes

        msgs_sent = jnp.minimum(msgs_sent + random_msgs_to_send, 1)  # [B, N, N]

        newly_updated_nodes = jnp.min(msgs_sent, axis=2)  # [B, N]

        current_min_step = jnp.expand_dims(jnp.min(updated_nodes, axis=-1), axis=-1)
        updated_nodes = jnp.where(newly_updated_nodes, current_min_step + 1, updated_nodes) # [B, N]
        newly_updated_nodes = updated_nodes - old_updated_nodes # [B, N]

        msgs_sent = jnp.where(jnp.expand_dims(newly_updated_nodes, -1), 0, msgs_sent)  # [B, N, N]

        updated_nodes_exp = jnp.expand_dims(newly_updated_nodes, -1)  # [B, N, 1]

        msg_1 = jnp.where(updated_nodes_exp, msg_1, old_msg1)  # [B, N, H]
        msg_2 = jnp.where(updated_nodes_exp, msg_2, old_msg2)  # [B, N, H]

        msgs_v0 = (jnp.expand_dims(old_msg1, axis=1) + jnp.expand_dims(old_msg2, axis=2) +
                   msg_e + jnp.expand_dims(msg_g, axis=(1, 2)))

        msgs_v1 = (jnp.expand_dims(old_msg1, axis=1) + jnp.expand_dims(msg_2, axis=2) +
                   msg_e + jnp.expand_dims(msg_g, axis=(1, 2)))

        msgs_v2 = (jnp.expand_dims(msg_1, axis=1) + jnp.expand_dims(old_msg2, axis=2) +
                   msg_e + jnp.expand_dims(msg_g, axis=(1, 2)))

        msgs_v3 = (jnp.expand_dims(msg_1, axis=1) + jnp.expand_dims(msg_2, axis=2) +
                   msg_e + jnp.expand_dims(msg_g, axis=(1, 2)))

        # v0 = jnp.equal(jnp.expand_dims(old_updated_nodes, axis=1), jnp.expand_dims(old_updated_nodes, axis=2))
        v1 = jnp.equal(jnp.expand_dims(old_updated_nodes, axis=1), jnp.expand_dims(updated_nodes, axis=2))
        v2 = jnp.equal(jnp.expand_dims(updated_nodes, axis=1), jnp.expand_dims(old_updated_nodes, axis=2))
        v3 = jnp.equal(jnp.expand_dims(updated_nodes, axis=1), jnp.expand_dims(updated_nodes, axis=2))
        msgs = jnp.where(jnp.expand_dims(v1, -1),
                         msgs_v1,
                         jnp.where(jnp.expand_dims(v2, -1),
                                   msgs_v2,
                                   jnp.where(jnp.expand_dims(v3, -1),
                                             msgs_v3,
                                             msgs_v0)))

        # msgs_to_send = jnp.expand_dims(jnp.multiply(adj_mat, jnp.expand_dims(newly_updated_nodes, 0)), -1)
        msgs_to_send = jnp.expand_dims(adj_mat, -1)

        # aggregation
        maxarg_sent = jnp.where(msgs_to_send, msgs, -BIG_NUMBER)

        acc_msgs_sent = jnp.max(maxarg_sent, axis=1)

        # update
        ret = jnp.maximum(node_fts, acc_msgs_sent)

        return ret, None, updated_nodes, msgs_sent, msg_1, msg_2  # pytype: disable=bad-return-type  # numpy-scalars