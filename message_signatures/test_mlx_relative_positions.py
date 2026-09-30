"""CPU-only checks for MLX DeBERTa position indexing without importing MLX."""
import ast
import unittest
from pathlib import Path

import numpy as np
import torch
from transformers.models.deberta_v2.modeling_deberta_v2 import make_log_bucket_position


MODULE = Path(__file__).with_name('mlx_deberta.py')


def load_numpy_index_helper():
    """Load the pure NumPy helper AST so tests never initialize an MLX device."""
    tree = ast.parse(MODULE.read_text(encoding='utf-8'))
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == '_relative_positions')
    isolated = ast.Module(body=[function], type_ignores=[])
    namespace = {'np': np}
    exec(compile(isolated, str(MODULE), 'exec'), namespace)
    return namespace['_relative_positions']


class RelativePositionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.indexer = staticmethod(load_numpy_index_helper())

    def test_c2p_and_transposed_p2c_indices_match_transformers(self):
        for length in (1, 2, 63, 127, 128, 129, 256, 384, 512):
            with self.subTest(length=length):
                relative = (torch.arange(length)[:, None] - torch.arange(length)[None, :])
                bucketed = make_log_bucket_position(relative, 256, 512).numpy()
                expected_c2p = np.clip(bucketed + 256, 0, 511).astype(np.int32)
                expected_p2c = np.clip(-bucketed + 256, 0, 511).astype(np.int32)
                observed_c2p, observed_p2c = self.indexer(length, 256, 512)
                np.testing.assert_array_equal(observed_c2p, expected_c2p)
                np.testing.assert_array_equal(observed_p2c, expected_p2c)

    def test_p2c_lookup_is_transposed_query_key_orientation(self):
        c2p, p2c = self.indexer(32, 256, 512)
        np.testing.assert_array_equal(p2c, c2p.T)


if __name__ == '__main__':
    unittest.main()
