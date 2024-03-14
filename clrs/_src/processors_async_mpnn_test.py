from absl.testing import absltest
import chex
from clrs._src import processors
from clrs._src import processors_async_mpnn
import haiku as hk
import jax
import jax.numpy as jnp
from jax import random
import numpy as np
from synjax._src.utils import semirings_dot_general
from synjax._src.utils import semirings_einsum

rng_state = np.random.RandomState(42)
rng = rng_state.randint(2**32)

class SemiringTest(absltest.TestCase):
    def test_einsum_log_semiring(self):
        def einsum_log(*operands, **kwargs):
            sum_fn = jax.nn.logsumexp
            mul_op = jnp.add
            dot_general = semirings_dot_general.build_dot_general(sum_fn, mul_op)
            return semirings_einsum.einsum_generalized(
                *operands, **kwargs,
                sum_fn=sum_fn, mul_op=mul_op, dot_general=dot_general)
        x = jax.random.uniform(jax.random.PRNGKey(0), (16, 32, 32))
        y = jax.random.uniform(jax.random.PRNGKey(0), (32, 16))
        expression = "...ij,jk->...ik"

        chex.assert_trees_all_close(
            jnp.log(jnp.einsum(expression, jnp.exp(x), jnp.exp(y))),
            einsum_log(expression, x, y),
            rtol=1e-3,
            atol=1e-3
        )

        chex.assert_trees_all_close(
            jnp.log(jnp.einsum(expression, jnp.exp(x), jnp.exp(y))),
            jnp.log(jnp.matmul(jnp.exp(x), jnp.exp(y)))
        )

    def test_stable_log_semiring(self):
        x = jax.random.uniform(jax.random.PRNGKey(0), (16, 32, 32))
        y = jax.random.uniform(jax.random.PRNGKey(0), (32, 16))

        def stable_log_semiring(inputs, w):
            inputs_max = jnp.max(inputs)
            w_max = jnp.max(w)
            out = jnp.log(jnp.matmul(jnp.exp(inputs - inputs_max), jnp.exp(w - w_max)))
            out = out + inputs_max + w_max
            return out

        chex.assert_trees_all_close(
            stable_log_semiring(x, y),
            jnp.log(jnp.matmul(jnp.exp(x), jnp.exp(y))),
            rtol=1e-3,
            atol=1e-3
        )

    def test_stable_log_sum(self):
        x = jax.random.uniform(jax.random.PRNGKey(0), (16, 32))
        y = jax.random.uniform(jax.random.PRNGKey(0), (16, 32))

        def stable_log_sum(a, b):
            maxval = max(jnp.max(a), jnp.max(b))
            out = jnp.log(jnp.exp(a - maxval) + jnp.exp(b - maxval)) + maxval
            return out

        chex.assert_trees_all_close(
            stable_log_sum(x, y),
            jnp.logaddexp(x, y),
            rtol=1e-3,
            atol=1e-3
        )


    def test_einsum_max_semiring(self):
        def einsum_max(*operands, **kwargs):
            sum_fn = jnp.max
            mul_op = jnp.add
            dot_general = semirings_dot_general.build_dot_general(sum_fn, mul_op)
            return semirings_einsum.einsum_generalized(
                *operands, **kwargs,
                sum_fn=sum_fn, mul_op=mul_op, dot_general=dot_general)
        x = jnp.array(
            [
                [0, 1, 2, 3],
                [-1, 0, 1, 2],
                [0, 1, 2, 3],
                [1, 2, 3, 4]
            ]
        )
        y = jnp.array(
            [
                [0, 1],
                [-1, 2],
                [-2, 3],
                [-3, 4]
            ]
        )
        expected_output = jnp.array(
            [
                [0, 7],
                [-1, 6],
                [0, 7],
                [1, 8]
            ]
        )
        expression = "...ij,jk->...ik"

        chex.assert_trees_all_close(
            expected_output,
            einsum_max(expression, x, y)
        )


class MPNNL1AsyncPropertiesTest(absltest.TestCase):

    def test_check_async_identical_to_l1(self):

        batch_size = 64
        out_size = 128
        use_ln = True
        nb_nodes = 4
        hidden_dim = 128

        def forward_fn_l1(node_fts, edge_fts, graph_fts, adj_mat, hidden):

            processor = processors.MPNNL1(
                out_size=out_size,
                msgs_mlp_sizes=[out_size, out_size],
                use_ln=use_ln,)
            return processor(
                node_fts=node_fts,
                edge_fts=edge_fts,
                graph_fts=graph_fts,
                adj_mat=adj_mat,
                hidden=hidden,
                batch_size=batch_size,
                nb_nodes=nb_nodes,)

        forward_l1 = hk.transform(forward_fn_l1)

        def forward_fn_l1_async(node_fts, edge_fts, graph_fts, adj_mat, hidden):

            processor = processors_async_mpnn.MPNNL1AsyncTest(
                out_size=out_size,
                msgs_mlp_sizes=[out_size, out_size],
                use_ln=use_ln,)
            return processor(
                node_fts=node_fts,
                edge_fts=edge_fts,
                graph_fts=graph_fts,
                adj_mat=adj_mat,
                hidden=hidden,
                batch_size=batch_size,
                nb_nodes=nb_nodes,)

        forward_l1_async = hk.transform(forward_fn_l1_async)

        key = random.PRNGKey(rng)

        # Initialise node/edge/graph features and adjacency matrix.
        node_fts = random.uniform(key, (batch_size, nb_nodes, hidden_dim))
        edge_fts = random.uniform(key, (batch_size, nb_nodes, nb_nodes, hidden_dim))
        graph_fts = random.uniform(key, (batch_size, hidden_dim))
        adj_mat = jnp.ones((batch_size, nb_nodes, nb_nodes))
        hidden = random.uniform(key, (batch_size, nb_nodes, hidden_dim))

        params_l1 = forward_l1.init(key, node_fts, edge_fts, graph_fts, adj_mat, hidden)
        params_l1_async = forward_l1_async.init(key, node_fts, edge_fts, graph_fts, adj_mat, hidden)
        ret_l1, _ = forward_l1.apply(
            params_l1, None, node_fts, edge_fts, graph_fts, adj_mat, hidden
        )
        ret_l1_async, _ = forward_l1_async.apply(
            params_l1_async, None, node_fts, edge_fts, graph_fts, adj_mat, hidden
        )

        chex.assert_type(ret_l1, jnp.float32)
        chex.assert_type(ret_l1_async, jnp.float32)
        chex.assert_equal_shape([ret_l1, ret_l1_async])
        chex.assert_trees_all_close(ret_l1, ret_l1_async)


class MPNNL2AsyncPropertiesTest(absltest.TestCase):

    def test_check_async_identical_to_l2(self):

        batch_size = 64
        out_size = 128
        nb_nodes = 4
        hidden_dim = 128

        def forward_fn_l2(node_fts, edge_fts, graph_fts, adj_mat, hidden):

            processor = processors.MPNNL2(
                out_size=out_size,)
            return processor(
                node_fts=node_fts,
                edge_fts=edge_fts,
                graph_fts=graph_fts,
                adj_mat=adj_mat,
                hidden=hidden,
                batch_size=batch_size,
                nb_nodes=nb_nodes,)

        forward_l2 = hk.transform(forward_fn_l2)

        def forward_fn_l2_async(node_fts, edge_fts, graph_fts, adj_mat, hidden, random_msgs_to_send, generate_msgs, msgs):

            processor = processors_async_mpnn.MPNNL2AsyncTest(
                out_size=out_size,)
            return processor(
                node_fts=node_fts,
                edge_fts=edge_fts,
                graph_fts=graph_fts,
                adj_mat=adj_mat,
                hidden=hidden,
                batch_size=batch_size,
                nb_nodes=nb_nodes,
                random_msgs_to_send=random_msgs_to_send,
                generate_msgs=generate_msgs,
                msgs=msgs,
            )

        forward_l2_async = hk.transform(forward_fn_l2_async)

        key = random.PRNGKey(rng)

        # Initialise node/edge/graph features and adjacency matrix.
        node_fts = random.uniform(key, (batch_size, nb_nodes, hidden_dim))
        edge_fts = random.uniform(key, (batch_size, nb_nodes, nb_nodes, hidden_dim))
        graph_fts = random.uniform(key, (batch_size, hidden_dim))
        adj_mat = jnp.ones((batch_size, nb_nodes, nb_nodes))
        hidden = random.uniform(key, (batch_size, nb_nodes, hidden_dim))

        params_l2 = forward_l2.init(key, node_fts, edge_fts, graph_fts, adj_mat, hidden)

        ret_l2, edge_l2 = forward_l2.apply(
            params_l2, None, node_fts, edge_fts, graph_fts, adj_mat, hidden
        )

        # randomly initialize random_msgs_to_send
        random_shape = jnp.expand_dims(adj_mat, -1).shape
        random_msgs_to_send = random.choice(key, jnp.array([0, 1]), shape=random_shape)

        params_l2_async = forward_l2_async.init(key, node_fts, edge_fts, graph_fts, adj_mat, hidden,
                                                random_msgs_to_send, True, None)

        new_node_fts, _, msgs, unsent_msgs = forward_l2_async.apply(
            params_l2_async,
            None,
            node_fts,
            edge_fts,
            graph_fts,
            adj_mat,
            hidden,
            random_msgs_to_send,
            generate_msgs=True,
            msgs=None,
        )

        ret_l2_async, _, _, _ = forward_l2_async.apply(
            params_l2_async,
            None,
            new_node_fts,
            edge_fts,
            graph_fts,
            adj_mat,
            hidden,
            random_msgs_to_send=unsent_msgs,
            generate_msgs=False,
            msgs=msgs
        )

        chex.assert_type(ret_l2, jnp.float32)
        chex.assert_type(ret_l2_async, jnp.float32)
        chex.assert_equal_shape([ret_l2, ret_l2_async])
        chex.assert_trees_all_close(ret_l2, ret_l2_async)


class MPNNL3AsyncPropertiesTest(absltest.TestCase):

    def test_check_async_identical_to_l3(self):

        batch_size = 10
        out_size = hidden_dim = 8
        nb_nodes = 8
        steps = 5
        BIG_NUMBER = 1e6

        key = random.PRNGKey(rng)

        # Initialise node/edge/graph features and adjacency matrix.
        node_fts = random.uniform(key, (batch_size, nb_nodes, hidden_dim))

        edge_fts = random.uniform(key, (batch_size, nb_nodes, nb_nodes, hidden_dim))

        graph_fts = random.uniform(key, (batch_size, hidden_dim))

        adj_mat = jnp.ones((batch_size, nb_nodes, nb_nodes))

        hidden = random.uniform(key, (batch_size, nb_nodes, hidden_dim))

        def forward_fn_l3(node_fts, edge_fts, graph_fts, adj_mat, hidden):

            processor = processors.MPNNL3(
                out_size=out_size,
                # semiring='maxplus',
            )
            return processor(
                node_fts=node_fts,
                edge_fts=edge_fts,
                graph_fts=graph_fts,
                adj_mat=adj_mat,
                hidden=hidden,
                batch_size=batch_size,
                nb_nodes=nb_nodes,)

        forward_l3 = hk.transform(forward_fn_l3)

        def forward_fn_l3_async(node_fts, edge_fts, graph_fts, adj_mat, hidden, updated_nodes, msgs_sent,
                                random_msgs_to_send, old_msg1, old_msg2):

            processor = processors_async_mpnn.MPNNL3AsyncTest(
                out_size=out_size,
                semiring='maxplus',
            )
            return processor(
                node_fts=node_fts,
                edge_fts=edge_fts,
                graph_fts=graph_fts,
                adj_mat=adj_mat,
                hidden=hidden,
                batch_size=batch_size,
                nb_nodes=nb_nodes,
                updated_nodes=updated_nodes,
                msgs_sent=msgs_sent,
                random_msgs_to_send=random_msgs_to_send,
                old_msg1=old_msg1,
                old_msg2=old_msg2,
            )

        forward_l3_async = hk.transform(forward_fn_l3_async)

        params_l3 = forward_l3.init(key, node_fts, edge_fts, graph_fts, adj_mat, hidden)

        ret_l3 = hidden
        for i in range(steps):
            ret_l3, _ = forward_l3.apply(
                params_l3, None, node_fts, edge_fts, graph_fts, adj_mat, ret_l3
            )

        random_msgs_to_send = random.choice(key, jnp.array([0, 1]), shape=(batch_size, nb_nodes, nb_nodes))

        updated_nodes = jnp.zeros((batch_size, nb_nodes))
        msgs_sent = jnp.zeros((batch_size, nb_nodes, nb_nodes))

        old_msg1 = jnp.full((batch_size, nb_nodes, out_size), -BIG_NUMBER)
        old_msg2 = jnp.full((batch_size, nb_nodes, out_size), -BIG_NUMBER)

        params_l3_async = forward_l3_async.init(key, node_fts, edge_fts, graph_fts, adj_mat, hidden,
                                                updated_nodes, msgs_sent, random_msgs_to_send, old_msg1, old_msg2)

        ret_l3_async = hidden
        newkey = key

        for i in range(2*steps):
            if i < steps:
                newkey, subkey = random.split(newkey)
                random_msgs_to_send = random.choice(newkey, jnp.array([0, 1]), shape=(batch_size, nb_nodes, nb_nodes))
            else:
                random_msgs_to_send = jnp.minimum(jnp.expand_dims(steps - updated_nodes, -1), 1 - msgs_sent)

            ret_l3_async, _, updated_nodes, msgs_sent, old_msg1, old_msg2 = forward_l3_async.apply(
                params_l3_async,
                None,
                node_fts,
                edge_fts,
                graph_fts,
                adj_mat,
                ret_l3_async,
                updated_nodes,
                msgs_sent,
                random_msgs_to_send,
                old_msg1,
                old_msg2,
            )

        chex.assert_type(ret_l3, jnp.float32)
        chex.assert_type(ret_l3_async, jnp.float32)
        chex.assert_equal_shape([ret_l3, ret_l3_async])
        chex.assert_trees_all_close(ret_l3, ret_l3_async)


if __name__ == '__main__':
    absltest.main()