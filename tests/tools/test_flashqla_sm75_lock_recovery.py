# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path


def _load_sm_legacy(monkeypatch):
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch_utils = types.ModuleType("torch.utils")
    cpp_extension = types.ModuleType("torch.utils.cpp_extension")
    cpp_extension.load = lambda *args, **kwargs: None

    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.utils", torch_utils)
    monkeypatch.setitem(sys.modules, "torch.utils.cpp_extension", cpp_extension)

    source = (
        Path(__file__).parents[2]
        / "tools"
        / "flashqla_sm75_patches"
        / "sm_legacy.py"
    )
    spec = importlib.util.spec_from_file_location("flashqla_sm75_sm_legacy", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_lock(root: Path, relative_parent: str) -> Path:
    lock = root / relative_parent / "flash_qla_legacy_gdn" / "lock"
    lock.parent.mkdir(parents=True)
    lock.touch()
    return lock


def test_recovers_orphans_in_direct_and_abi_scoped_layouts(tmp_path, monkeypatch):
    module = _load_sm_legacy(monkeypatch)
    direct = _make_lock(tmp_path, "")
    nested = _make_lock(tmp_path, "py312_cu130")
    unrelated = tmp_path / "other_extension" / "lock"
    unrelated.parent.mkdir()
    unrelated.touch()

    recovered = module._recover_orphan_extension_locks(
        tmp_path,
        stale_after_seconds=0,
        recheck_delay_seconds=0,
    )

    assert {path.parent for path in recovered} == {direct.parent, nested.parent}
    assert not direct.exists()
    assert not nested.exists()
    assert len(list(direct.parent.glob("lock.orphan-*"))) == 1
    assert len(list(nested.parent.glob("lock.orphan-*"))) == 1
    assert unrelated.exists()


def test_preserves_lock_while_a_process_holds_its_inode(tmp_path, monkeypatch):
    module = _load_sm_legacy(monkeypatch)
    lock = _make_lock(tmp_path, "")
    fd = os.open(lock, os.O_RDONLY)
    try:
        recovered = module._recover_orphan_extension_locks(
            tmp_path,
            stale_after_seconds=0,
            recheck_delay_seconds=0,
        )
    finally:
        os.close(fd)

    assert recovered == []
    assert lock.exists()
    assert not list(lock.parent.glob("lock.orphan-*"))


def test_preserves_lock_during_the_staleness_grace_period(tmp_path, monkeypatch):
    module = _load_sm_legacy(monkeypatch)
    lock = _make_lock(tmp_path, "")

    recovered = module._recover_orphan_extension_locks(
        tmp_path,
        stale_after_seconds=30,
        recheck_delay_seconds=0,
    )

    assert recovered == []
    assert lock.exists()
