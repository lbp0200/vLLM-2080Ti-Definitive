import importlib.util
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).parents[2] / "tools" / "patch_torch_inductor_e8m0.py"
SPEC = importlib.util.spec_from_file_location("patch_torch_inductor_e8m0", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
PATCHER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PATCHER)


class TestPatchTorchInductorE8M0(unittest.TestCase):
    def write_lowering(self, source: str) -> Path:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        path = Path(temp_dir.name) / "lowering.py"
        path.write_text(source, encoding="utf-8")
        return path

    def test_applies_fix_once(self):
        path = self.write_lowering(
            PATCHER.FUNCTION + PATCHER.FIRST_STATEMENT + "    return True\n"
        )

        self.assertTrue(PATCHER.patch_lowering(path))
        self.assertFalse(PATCHER.patch_lowering(path))
        self.assertEqual(path.read_text(encoding="utf-8").count(PATCHER.FIX), 1)

    def test_rejects_unknown_layout(self):
        path = self.write_lowering(PATCHER.FUNCTION + "    return True\n")

        with self.assertRaisesRegex(ValueError, "validated 2.13.0 layout"):
            PATCHER.patch_lowering(path)

    def test_rejects_missing_function(self):
        path = self.write_lowering("def unrelated():\n    return True\n")

        with self.assertRaisesRegex(ValueError, "found 0"):
            PATCHER.patch_lowering(path)


if __name__ == "__main__":
    unittest.main()
