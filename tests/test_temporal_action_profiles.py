import importlib.util
import math
import struct
import tempfile
import unittest
from pathlib import Path


SPEC = importlib.util.spec_from_file_location(
    "action_profiles", Path(__file__).resolve().parents[1] / "scripts/verify_temporal_action_profiles.py")
VERIFY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFY)


class PhysicalEnergyTests(unittest.TestCase):
    def test_energy_uses_physical_hop_width_and_actual_routing(self):
        self.assertEqual(VERIFY.check_energy([[5., 4.], [0., 0.]],
                         [1., 3., -2., 2.], ["owned", None], 2, "prefix"), (4, 0.))
        values = [1.] * 480 + [2.] * 32
        expected = (480 + 32 * 4) / 512
        self.assertEqual(VERIFY.check_energy([[expected], [expected]],
                         values, ["owned", "owned"], 512, "future"), (2, 0.))
        with self.assertRaisesRegex(ValueError, "energy mismatch"):
            VERIFY.check_energy([[1.], [1.]], values, ["owned", "owned"], 512, "future")

    def test_missing_partial_nonfinite_and_routed_away_values_fail(self):
        for reported in [[[5., 4.]], [[5.], [0., 0.]], [[5., 4.], [5., 4.]],
                         [[math.nan, 4.], [0., 0.]], [[math.inf, 4.], [0., 0.]]]:
            with self.assertRaises(ValueError):
                VERIFY.check_energy(reported, [1., 3., -2., 2.], ["owned", None], 2, "prefix")
        with self.assertRaisesRegex(ValueError, "wrong energy grid"):
            VERIFY.check_energy([[5.], [0.]], [1., 3., -2.], ["owned", None], 2, "prefix")

    def test_pcm_requires_the_exact_registered_finite_support(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "prefix.f32le"
            path.write_bytes(struct.pack("<4f", 1., -2., 0., 0.5))
            pcm, samples = VERIFY.read_pcm(path, 4)
            self.assertEqual(len(pcm), 16)
            self.assertEqual(samples, (1., -2., 0., 0.5))
            for count in [3, 5]:
                with self.assertRaisesRegex(ValueError, "incomplete PCM support"):
                    VERIFY.read_pcm(path, count)
            path.write_bytes(struct.pack("<4f", 1., math.nan, 0., 0.5))
            with self.assertRaisesRegex(ValueError, "nonfinite PCM"):
                VERIFY.read_pcm(path, 4)


if __name__ == "__main__":
    unittest.main()
