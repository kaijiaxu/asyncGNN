from absl.testing import absltest
import chex
from clrs._src import processors
from clrs._src import processors_async_gat
import haiku as hk
import jax
import jax.numpy as jnp
from jax import random
import numpy as np

rng_state = np.random.RandomState(42)
rng = rng_state.randint(2**32)


class GATL1Test(absltest.TestCase):

    def test_check_identical_to_baseline(self):

        batch_size = 64
        out_size = 128
        use_ln = True
        nb_nodes = 4
        hidden_dim = 128
        nb_heads = 2

        def forward_fn_baseline(node_fts, edge_fts, graph_fts, adj_mat, hidden):

            processor = processors.GAT(
                out_size=out_size,
                nb_heads=nb_heads,
                use_ln=use_ln,)
            return processor(
                node_fts=node_fts,
                edge_fts=edge_fts,
                graph_fts=graph_fts,
                adj_mat=adj_mat,
                hidden=hidden,
                batch_size=batch_size,
                nb_nodes=nb_nodes,)

        forward_baseline = hk.transform(forward_fn_baseline)

        def forward_fn_l1(node_fts, edge_fts, graph_fts, adj_mat, hidden):

            processor = processors.GATL1(
                out_size=out_size,
                nb_heads=nb_heads,
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

        key = random.PRNGKey(rng)

        # Initialise node/edge/graph features and adjacency matrix.
        node_fts = random.uniform(key, (batch_size, nb_nodes, hidden_dim))
        edge_fts = random.uniform(key, (batch_size, nb_nodes, nb_nodes, hidden_dim))
        graph_fts = random.uniform(key, (batch_size, hidden_dim))
        adj_mat = jnp.repeat(
            jnp.expand_dims(jnp.eye(nb_nodes), 0), batch_size, axis=0)
        for _ in range(1000):
            key, subkey = random.split(key)
            b = random.randint(subkey, (), 0, batch_size)
            n1 = random.randint(subkey, (), 0, nb_nodes)
            key, subkey = random.split(key)
            n2 = random.randint(subkey, (), 0, nb_nodes)
            adj_mat = adj_mat.at[b, n1, n2].set(1)
        hidden = random.uniform(key, (batch_size, nb_nodes, hidden_dim))

        params_baseline = forward_baseline.init(key, node_fts, edge_fts, graph_fts, adj_mat, hidden)
        params_l1 = forward_l1.init(key, node_fts, edge_fts, graph_fts, adj_mat, hidden)

        ret_baseline, tri_msgs_baseline = forward_baseline.apply(
            params_baseline, None, node_fts, edge_fts, graph_fts, adj_mat, hidden
        )
        ret_l1, tri_msgs_l1 = forward_l1.apply(
            params_l1, None, node_fts, edge_fts, graph_fts, adj_mat, hidden
        )
        chex.assert_type(ret_l1, jnp.float32)
        chex.assert_equal_shape([ret_l1, ret_baseline])
        chex.assert_trees_all_close(ret_l1, ret_baseline, atol=1e-06)
        chex.assert_equal(tri_msgs_l1, None)


class GATL1AsyncTest(absltest.TestCase):

    def test_check_identical_to_l1(self):

        batch_size = 64
        out_size = 128
        use_ln = True
        nb_nodes = 4
        hidden_dim = 128
        nb_heads = 2

        def forward_fn_l1(node_fts, edge_fts, graph_fts, adj_mat, hidden):

            processor = processors.GATL1(
                out_size=out_size,
                nb_heads=nb_heads,
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

            processor_async = processors_async_gat.GATL1AsyncTest(
                out_size=out_size,
                nb_heads=nb_heads,
                use_ln=use_ln,)
            return processor_async(
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
        adj_mat = jnp.repeat(
            jnp.expand_dims(jnp.eye(nb_nodes), 0), batch_size, axis=0)
        hidden = random.uniform(key, (batch_size, nb_nodes, hidden_dim))
        # randomly initialize adj_mat
        for _ in range(1000):
            key, subkey = random.split(key)
            b = random.randint(subkey, (), 0, batch_size)
            n1 = random.randint(subkey, (), 0, nb_nodes)
            key, subkey = random.split(key)
            n2 = random.randint(subkey, (), 0, nb_nodes)
            adj_mat = adj_mat.at[b, n1, n2].set(1)
        params_l1 = forward_l1.init(key, node_fts, edge_fts, graph_fts, adj_mat, hidden)
        params_l1_async = forward_l1_async.init(key, node_fts, edge_fts, graph_fts, adj_mat, hidden)

        ret_l1, tri_msgs_l1 = forward_l1.apply(
            params_l1, None, node_fts, edge_fts, graph_fts, adj_mat, hidden
        )
        ret_l1_async, tri_msgs_l1_async = forward_l1_async.apply(
            params_l1_async, None, node_fts, edge_fts, graph_fts, adj_mat, hidden
        )
        chex.assert_type(ret_l1_async, jnp.float32)
        chex.assert_equal_shape([ret_l1_async, ret_l1])
        chex.assert_trees_all_close(ret_l1_async, ret_l1, atol=1e-06)
        chex.assert_equal(tri_msgs_l1_async, None)


class GATv2L1AsyncTest(absltest.TestCase):

    def test_check_identical_to_l1(self):

        batch_size = 64
        out_size = 128
        use_ln = True
        nb_nodes = 4
        hidden_dim = 128
        nb_heads = 2

        def forward_fn_l1(node_fts, edge_fts, graph_fts, adj_mat, hidden):

            processor = processors.GATv2L1(
                out_size=out_size,
                nb_heads=nb_heads,
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

            processor_async = processors_async_gat.GATv2L1AsyncTest(
                out_size=out_size,
                nb_heads=nb_heads,
                use_ln=use_ln,)
            return processor_async(
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
        adj_mat = jnp.repeat(
            jnp.expand_dims(jnp.eye(nb_nodes), 0), batch_size, axis=0)
        hidden = random.uniform(key, (batch_size, nb_nodes, hidden_dim))
        # randomly initialize adj_mat
        for _ in range(1000):
            key, subkey = random.split(key)
            b = random.randint(subkey, (), 0, batch_size)
            n1 = random.randint(subkey, (), 0, nb_nodes)
            key, subkey = random.split(key)
            n2 = random.randint(subkey, (), 0, nb_nodes)
            adj_mat = adj_mat.at[b, n1, n2].set(1)
        # adj_mat = jnp.ones((batch_size, nb_nodes, nb_nodes))
        params_l1 = forward_l1.init(key, node_fts, edge_fts, graph_fts, adj_mat, hidden)
        params_l1_async = forward_l1_async.init(key, node_fts, edge_fts, graph_fts, adj_mat, hidden)

        ret_l1, tri_msgs_l1 = forward_l1.apply(
            params_l1, None, node_fts, edge_fts, graph_fts, adj_mat, hidden
        )
        ret_l1_async, tri_msgs_l1_async = forward_l1_async.apply(
            params_l1_async, None, node_fts, edge_fts, graph_fts, adj_mat, hidden
        )
        chex.assert_type(ret_l1_async, jnp.float32)
        chex.assert_equal_shape([ret_l1_async, ret_l1])
        chex.assert_trees_all_close(list(params_l1.values()), list(params_l1_async.values()))
        chex.assert_trees_all_close(ret_l1_async, ret_l1, atol=1e-06)
        chex.assert_equal(tri_msgs_l1_async, None)


if __name__ == '__main__':
    absltest.main()