from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from discogen import create_discobench
from discogen.utils import get_discobench_tasks


@pytest.fixture
def mock_make_files() -> Iterator[MagicMock]:
    """Mock make_files as this is tested separately, and we don't want to deal with making new files here."""
    # Patch the class itself. We yield the class mock 'mocked' so we can
    # assert the constructor was called.
    with patch("discogen.create_discobench.MakeFiles") as mocked:
        yield mocked


@pytest.mark.parametrize("task_name", get_discobench_tasks())
@pytest.mark.parametrize("eval_type", ["performance", "energy", "time", "golf", "memory"])
@pytest.mark.parametrize("test_mode, expected_train", [(True, False), (False, True)])
def test_create_discobench_task(
    mock_make_files: MagicMock, test_mode: bool, expected_train: bool, eval_type: str, task_name: str
) -> None:
    """Tests that each discobench task can be created."""
    create_discobench(task_name, test=test_mode, eval_type=eval_type, no_data=True)

    with open(f"discogen/discobench_configs/{task_name}.yaml") as f:
        task_config = yaml.safe_load(f)

    mock_make_files.return_value.make_files.assert_called_with(
        task_config, train=expected_train, use_base=False, no_data=True, baseline_scale=1.0, eval_type=eval_type
    )


@pytest.mark.parametrize("task_name", ["GreenhouseGasPrediction_all", "OnPolicyRL_train", "OffPolicyRL_networks"])
@pytest.mark.parametrize("source_path", ["tmp1", "tmp2", None])
def test_different_source_path(
    task_name: str, source_path: str | None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tests that each discobench task can be created."""
    # Run inside a temp dir so generated files (task_src/, source_path, cache/) don't pollute the cwd.
    monkeypatch.chdir(tmp_path)
    create_discobench(task_name, test=False, no_data=True, source_path=source_path)

    task_name_sp = Path(f"task_src/{task_name}")
    if source_path is not None:
        sp = Path(source_path)
        assert sp.exists()
        assert not task_name_sp.exists()
    else:
        assert task_name_sp.exists()
