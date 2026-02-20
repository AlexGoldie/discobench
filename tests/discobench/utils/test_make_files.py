"""Tests for MakeFiles class, parameterised over all domains from discobench."""

import os
from pathlib import Path
from typing import Any

import pytest
import yaml

from discobench import create_config
from discobench.utils import get_domains
from discobench.utils.make_files import MakeFiles

ALL_DOMAINS = get_domains()


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# Fixtures
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


@pytest.fixture(params=ALL_DOMAINS, ids=ALL_DOMAINS)
def domain(request: pytest.FixtureRequest) -> str:
    """Return each domain from get_domains() as a parameterised fixture."""
    return str(request.param)


@pytest.fixture
def mf(domain: str) -> MakeFiles:
    """Return a real MakeFiles instance for the given domain."""
    return MakeFiles(domain)


@pytest.fixture
def config(domain: str) -> dict[str, Any]:
    """Return a valid config for the given domain."""
    return create_config(domain)


@pytest.fixture
def example_config(domain: str) -> dict[str, Any]:
    """Return the example config, with disjoint meta-train/meta-test, for the given domain."""
    conf_path = f"discobench/example_configs/{domain}.yaml"
    with open(conf_path) as f:
        config: dict[str, Any] = yaml.safe_load(f)
    return config


@pytest.fixture
def source_path(tmp_path: Path) -> Path:
    """Return a clean tmp source_path directory."""
    sp = tmp_path / "task_src"
    sp.mkdir()
    return sp


@pytest.fixture
def config_with_tmp(config: dict[str, Any], tmp_path: Path) -> dict[str, Any]:
    """Return a config whose source_path points to tmp_path."""
    cfg = dict(config)
    cfg["source_path"] = str(tmp_path / "task_src")
    return cfg


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# Tests for _setup_source_directory
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


class TestSetupSourceDirectory:
    """All tests for _setup_source_directory."""

    def test_train_wipes_entire_directory(self, mf: MakeFiles, source_path: Path) -> None:
        """Test that using train=True wipes the source directory."""
        sp = source_path

        (sp / "abc.txt").write_text("123")
        (sp / "discovered").mkdir()
        (sp / "discovered" / "network.py").write_text("xyz")

        mf.source_path = sp
        mf._setup_source_directory(train=True)
        assert not sp.exists()

    def test_test_preserves_discovered(self, mf: MakeFiles, source_path: Path) -> None:
        """Test that using train=False wipes the source directory *except* the discovered/ file."""
        sp = source_path

        (sp / "abc.txt").write_text("123")
        subdir = sp / "subdir"
        subdir.mkdir()
        (subdir / "inner.txt").write_text("inner")
        (sp / "discovered").mkdir()
        (sp / "discovered" / "network.py").write_text("xyz")

        mf.source_path = sp
        mf._setup_source_directory(train=False)

        assert (sp / "discovered").exists()
        assert (sp / "discovered/network.py").read_text() == "xyz"
        assert not (sp / "abc.txt").exists()
        assert not subdir.exists()


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# Tests for _normalize_task_ids
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


class TestNormalizeTaskIds:
    """All tests for _normalize_task_ids."""

    @pytest.mark.parametrize("train_test", ["train", "test"])
    @pytest.mark.parametrize("use_list", [True, False])
    def test_task_ids(self, mf: MakeFiles, example_config: dict[str, Any], train_test: str, use_list: bool) -> None:
        """Test that task_ids are always returned as a list, whether input is a list or scalar."""
        # TODO: use_list=False still wraps in a list; never tests bare scalar input,
        # so the `if not isinstance(task_ids, list)` branch in _normalize_task_ids is uncovered.
        if not use_list:
            example_config["train_task_id"] = [example_config["train_task_id"][0]]
            example_config["test_task_id"] = [example_config["test_task_id"][0]]

        task_ids = mf._normalize_task_ids(example_config, train_test)

        if train_test == "train":
            assert task_ids == example_config["train_task_id"]
        elif train_test == "test":
            assert task_ids == example_config["test_task_id"]

    def test_missing_key_raises(self, mf: MakeFiles) -> None:
        """Test that a missing task_id key raises KeyError."""
        with pytest.raises(KeyError):
            mf._normalize_task_ids({}, "train")


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# Tests for _normalize_model_ids
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


class TestNormalizeModelIds:
    """All tests for _normalize_model_ids."""

    @pytest.mark.parametrize("train_test", ["train", "test"])
    @pytest.mark.parametrize("use_list", [True, False])
    def test_no_model_ids_returns_nones(
        self, mf: MakeFiles, example_config: dict[str, Any], train_test: str, use_list: bool
    ) -> None:
        """Test that model_id defaults to [None, ...] when not set in the config."""
        # TODO: skips (and reports as passed) for domains that have model IDs
        # (e.g. ModelUnlearning). Intentional?
        if f"{train_test}_model_id" not in example_config:
            model_ids = mf._normalize_model_ids(example_config, train_test, example_config[f"{train_test}_task_id"])
            assert model_ids == [None] * len(example_config[f"{train_test}_task_id"])

    @pytest.mark.parametrize("mode", ["train", "test"])
    @pytest.mark.parametrize(
        ("task_id", "model_id", "expected_outcome"),
        [
            (["123"], ["abc"], ["abc"]),
            (["123", "456", "789"], ["abc", "def", "ghi"], ["abc", "def", "ghi"]),
            (["123", "456"], ["abc", "def", "ghi"], "Length of"),
            (["123", "456", "789"], ["abc", "def"], "Length of"),
            (["123", "456"], ["abc"], ["abc", "abc"]),
            (["123", "456"], "abc", ["abc", "abc"]),
            (["123", "456"], 123, "string or list"),
        ],
    )
    def test_different_models(
        self,
        mf: MakeFiles,
        example_config: dict[str, Any],
        task_id: list[str],
        model_id: list[str] | str | int,
        mode: str,
        expected_outcome: list[str] | str,
    ) -> None:
        """Test model_id output is always the right length, or raises when appropriate."""
        example_config[f"{mode}_task_id"] = task_id
        example_config[f"{mode}_model_id"] = model_id

        if isinstance(expected_outcome, str):
            with pytest.raises(ValueError, match=expected_outcome):
                mf._normalize_model_ids(example_config, mode, task_id)
        else:
            actual_outcome = mf._normalize_model_ids(example_config, mode, task_id)
            assert expected_outcome == actual_outcome


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# Tests for _build_base_description
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


class TestBuildBaseDescription:
    """All tests for _build_base_description."""

    def test_returns_nonempty_string(self, mf: MakeFiles, config: dict[str, Any]) -> None:
        """Test that the base description combines discobench and domain descriptions."""
        template_backend = config.get("template_backend", "default")
        result = mf._build_base_description(template_backend)
        assert isinstance(result, str)
        assert len(result) > 0


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# Tests for _load_model_description
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


class TestLoadModelDescription:
    """All tests for _load_model_description."""

    def test_returns_description_when_exists(self, mf: MakeFiles, tmp_path: Path) -> None:
        """Test that the description is returned when description.md exists."""
        model_path = tmp_path / "model_x"
        model_path.mkdir()
        (model_path / "description.md").write_text("Model X description")
        assert mf._load_model_description(model_path) == "Model X description"

    def test_returns_empty_when_missing(self, mf: MakeFiles, tmp_path: Path) -> None:
        """Test that an empty string is returned when description.md is missing."""
        model_path = tmp_path / "model_y"
        model_path.mkdir()
        assert mf._load_model_description(model_path) == ""


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# Tests for _copy_model_files
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


class TestCopyModelFiles:
    """All tests for _copy_model_files."""

    def test_copies_files_excluding_description(self, mf: MakeFiles, tmp_path: Path) -> None:
        """Test that all files except description.md are copied to the destination."""
        model_path = tmp_path / "model"
        model_path.mkdir()
        (model_path / "description.md").write_text("desc")
        (model_path / "weights.bin").write_bytes(b"\x00\x01")
        (model_path / "config.json").write_text("{}")

        dest = tmp_path / "dest"
        dest.mkdir()

        mf._copy_model_files(model_path, dest)
        assert (dest / "weights.bin").exists()
        assert (dest / "config.json").exists()
        assert not (dest / "description.md").exists()

    def test_copies_subdirectories(self, mf: MakeFiles, tmp_path: Path) -> None:
        """Test that subdirectories within the model path are recursively copied."""
        model_path = tmp_path / "model"
        sub = model_path / "layers"
        sub.mkdir(parents=True)
        (sub / "layer1.py").write_text("code")
        (model_path / "description.md").write_text("desc")

        dest = tmp_path / "dest"
        dest.mkdir()

        mf._copy_model_files(model_path, dest)
        assert (dest / "layers" / "layer1.py").read_text() == "code"

    def test_raises_if_path_missing(self, mf: MakeFiles, tmp_path: Path) -> None:
        """Test that ValueError is raised when the model path does not exist."""
        with pytest.raises(ValueError, match="model_path does not exist"):
            mf._copy_model_files(tmp_path / "nonexistent", tmp_path)


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# Tests for _process_single_task
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


class TestProcessSingleTask:
    """All tests for _process_single_task."""

    def test_processes_first_train_task(self, mf: MakeFiles, config: dict[str, Any], tmp_path: Path) -> None:
        """Test that processing a single task creates fixed files and returns descriptions."""
        task_ids = mf._normalize_task_ids(config, "train")
        model_ids = mf._normalize_model_ids(config, "train", task_ids)
        template_backend = config.get("template_backend", "default")
        sp = tmp_path / "task_src"
        (sp / "discovered").mkdir(parents=True, exist_ok=True)
        cfg = dict(config)
        cfg["source_path"] = str(sp)

        discovered, data_desc, model_desc = mf._process_single_task(
            task_id=task_ids[0],
            model_id=model_ids[0],
            config=cfg,
            train_test="train",
            template_backend=template_backend,
            train=True,
            use_base=True,
            no_data=True,
        )

        assert isinstance(discovered, list)
        assert isinstance(data_desc, str)
        assert len(data_desc) > 0
        assert isinstance(model_desc, str)

        dest = sp / f"{task_ids[0]}_{model_ids[0]}" if model_ids[0] is not None else sp / task_ids[0]

        for fixed_file in mf.task_spec["fixed_files"]:
            assert (dest / fixed_file).exists(), f"Fixed file '{fixed_file}' missing from {dest}"


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# Tests for _build_full_description
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


class TestBuildFullDescription:
    """All tests for _build_full_description."""

    def test_includes_all_parts(self, mf: MakeFiles) -> None:
        """Test that the full description includes base, prompts, data, and model descriptions."""
        result = mf._build_full_description(
            base_description="Base.",
            all_discovered_files=["model.py"],
            data_descriptions=["Data desc 0"],
            model_descriptions=["Model desc 0"],
            task_information={"model_prompt": "Improve it."},
        )
        assert "Base." in result
        assert "Improve it." in result
        assert "Data desc 0" in result
        assert "Model desc 0" in result
        assert "Problem 0" in result

    def test_no_prompt_for_unknown_file(self, mf: MakeFiles) -> None:
        """Test that a discovered file without a matching prompt key adds nothing."""
        result = mf._build_full_description(
            base_description="Base.",
            all_discovered_files=["unknown.py"],
            data_descriptions=["Data"],
            model_descriptions=[""],
            task_information={"model_prompt": "Only for model."},
        )
        assert "Only for model." not in result

    def test_multiple_data_descriptions(self, mf: MakeFiles) -> None:
        """Test that multiple tasks produce numbered Problem sections."""
        result = mf._build_full_description(
            base_description="Base.",
            all_discovered_files=[],
            data_descriptions=["Desc A", "Desc B"],
            model_descriptions=["", ""],
            task_information={},
        )
        assert "Problem 0" in result
        assert "Problem 1" in result
        assert "Desc A" in result
        assert "Desc B" in result


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# Tests for _create_symlinks_for_discovered
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


class TestCreateSymlinksForDiscovered:
    """All tests for _create_symlinks_for_discovered."""

    def test_creates_links_without_model_ids(self, mf: MakeFiles, tmp_path: Path) -> None:
        """Test that symlinks are created for each task when no model_ids are used."""
        mf.source_path = tmp_path / "src"
        disc = mf.source_path / "discovered"
        disc.mkdir(parents=True)
        (disc / "model.py").write_text("code")
        (mf.source_path / "t1").mkdir()
        (mf.source_path / "t2").mkdir()

        mf._create_symlinks_for_discovered(["model.py"], ["t1", "t2"], [None, None])
        assert (mf.source_path / "t1" / "model.py").is_symlink()
        assert (mf.source_path / "t2" / "model.py").is_symlink()

    def test_creates_links_with_model_ids(self, mf: MakeFiles, tmp_path: Path) -> None:
        """Test that symlinks use {task_id}_{model_id} directory naming."""
        mf.source_path = tmp_path / "src"
        disc = mf.source_path / "discovered"
        disc.mkdir(parents=True)
        (disc / "model.py").write_text("code")

        mf._create_symlinks_for_discovered(["model.py"], ["t1"], ["m1"])
        assert (mf.source_path / "t1_m1" / "model.py").is_symlink()


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# Tests for _create_fixed
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


class TestCreateFixed:
    """All tests for _create_fixed."""

    def test_copies_real_fixed_files(self, mf: MakeFiles, config: dict[str, Any], tmp_path: Path) -> None:
        """Test that all fixed files from the task_spec are created in the destination."""
        task_ids = mf._normalize_task_ids(config, "train")
        template_backend = config.get("template_backend", "default")
        task_path = mf.base_path / "datasets" / task_ids[0]
        dest = tmp_path / "dest"
        dest.mkdir()

        for fixed_file in mf.task_spec["fixed_files"]:
            mf._create_fixed(fixed_file, task_path, dest, template_backend)
            created = dest / fixed_file
            assert created.exists(), f"Fixed file '{fixed_file}' was not created."


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# Tests for _create_editable
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


class TestCreateEditable:
    """All tests for _create_editable."""

    def test_no_change_copies_base_to_dest(self, mf: MakeFiles, config: dict[str, Any], tmp_path: Path) -> None:
        """Test that change=False copies the base template into the task destination."""
        task_ids = mf._normalize_task_ids(config, "train")
        template_backend = config.get("template_backend", "default")
        task_path = mf.base_path / "datasets" / task_ids[0]
        dest = tmp_path / "dest"
        dest.mkdir()
        mf.source_path = tmp_path / "src"
        mf.source_path.mkdir()

        for module_file in mf.task_spec["module_files"]:
            mf._create_editable(
                module_file, task_path, dest, change=False, template_backend=template_backend, train=True, use_base=True
            )
            assert (dest / module_file).exists()

    def test_change_train_copies_to_discovered(self, mf: MakeFiles, config: dict[str, Any], tmp_path: Path) -> None:
        """Test that change=True during training copies the template into discovered/."""
        task_ids = mf._normalize_task_ids(config, "train")
        template_backend = config.get("template_backend", "default")
        task_path = mf.base_path / "datasets" / task_ids[0]
        dest = tmp_path / "dest"
        dest.mkdir()
        mf.source_path = tmp_path / "src"
        discovered = mf.source_path / "discovered"
        discovered.mkdir(parents=True)

        for module_file in mf.task_spec["module_files"]:
            mf._create_editable(
                module_file, task_path, dest, change=True, template_backend=template_backend, train=True, use_base=True
            )
            assert (discovered / module_file).exists()

    def test_change_test_skips(self, mf: MakeFiles, config: dict[str, Any], tmp_path: Path) -> None:
        """Test that change=True during test does not create any files."""
        task_ids = mf._normalize_task_ids(config, "train")
        template_backend = config.get("template_backend", "default")
        task_path = mf.base_path / "datasets" / task_ids[0]
        dest = tmp_path / "dest"
        dest.mkdir()
        mf.source_path = tmp_path / "src"
        discovered = mf.source_path / "discovered"
        discovered.mkdir(parents=True)

        for module_file in mf.task_spec["module_files"]:
            mf._create_editable(
                module_file, task_path, dest, change=True, template_backend=template_backend, train=False, use_base=True
            )
            assert not (discovered / module_file).exists()
            assert not (dest / module_file).exists()


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# Tests for _create_sym_link
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


class TestCreateSymLink:
    """All tests for _create_sym_link."""

    def test_creates_relative_symlink(self, mf: MakeFiles, tmp_path: Path) -> None:
        """Test that a relative symlink is created pointing from task dir to discovered/."""
        mf.source_path = tmp_path / "src"
        discovered = mf.source_path / "discovered"
        discovered.mkdir(parents=True)
        (discovered / "model.py").write_text("code")

        task_dir = mf.source_path / "task_a"
        task_dir.mkdir(parents=True)

        mf._create_sym_link("model.py", "task_a")

        link = task_dir / "model.py"
        assert link.is_symlink()
        assert link.resolve() == (discovered / "model.py").resolve()
        assert link.read_text() == "code"

    def test_replaces_existing_file(self, mf: MakeFiles, tmp_path: Path) -> None:
        """Test that an existing file at the destination is replaced by a symlink."""
        mf.source_path = tmp_path / "src"
        discovered = mf.source_path / "discovered"
        discovered.mkdir(parents=True)
        (discovered / "model.py").write_text("new code")

        task_dir = mf.source_path / "task_a"
        task_dir.mkdir(parents=True)
        (task_dir / "model.py").write_text("old")

        mf._create_sym_link("model.py", "task_a")

        link = task_dir / "model.py"
        assert link.is_symlink()
        assert link.read_text() == "new code"

    def test_no_master_file_does_nothing(self, mf: MakeFiles, tmp_path: Path) -> None:
        """Test that no symlink is created when the master file does not exist."""
        mf.source_path = tmp_path / "src"
        discovered = mf.source_path / "discovered"
        discovered.mkdir(parents=True)

        task_dir = mf.source_path / "task_a"
        task_dir.mkdir(parents=True)

        mf._create_sym_link("model.py", "task_a")
        assert not (task_dir / "model.py").exists()


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# Tests for _load_domain_description
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


class TestLoadDomainDescription:
    """All tests for _load_domain_description."""

    def test_default_description_exists(self, mf: MakeFiles) -> None:
        """Test that the default domain description loads and is non-empty."""
        result = mf._load_domain_description("default")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_falls_back_to_utils_for_nonexistent_backend(self, mf: MakeFiles) -> None:
        """Test that a nonexistent backend falls back to utils/description.md."""
        result = mf._load_domain_description("nonexistent_backend_xyz")
        expected = (mf.base_path / "utils" / "description.md").read_text(encoding="utf-8")
        assert result == expected


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# Tests for _load_domain_task_information
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


class TestLoadDomainTaskInformation:
    """All tests for _load_domain_task_information."""

    def test_loads_task_information(self, mf: MakeFiles, config: dict[str, Any]) -> None:
        """Test that task information loads as a non-empty dict."""
        template_backend = config.get("template_backend", "default")
        result = mf._load_domain_task_information(template_backend)
        assert isinstance(result, dict)

    def test_falls_back_to_utils_for_nonexistent_backend(self, mf: MakeFiles) -> None:
        """Test that a nonexistent backend falls back to utils/task_information.yaml."""
        result = mf._load_domain_task_information("nonexistent_backend_xyz")
        expected_path = mf.base_path / "utils" / "task_information.yaml"
        with open(expected_path) as f:
            expected = yaml.safe_load(f)
        assert result == expected


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# Tests for _get_data_description
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


class TestGetDataDescription:
    """All tests for _get_data_description."""

    def test_real_datasets_have_descriptions(self, mf: MakeFiles, config: dict[str, Any]) -> None:
        """Test that every train dataset has a non-empty description.md."""
        task_ids = mf._normalize_task_ids(config, "train")
        for task_id in task_ids:
            task_path = mf.base_path / "datasets" / task_id
            result = mf._get_data_description(task_path)
            assert isinstance(result, str)
            assert len(result) > 0


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# Tests for _save_description
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


class TestSaveDescription:
    """All tests for _save_description."""

    def test_writes_file(self, mf: MakeFiles, source_path: Path) -> None:
        """Test that _save_description writes the description to description.md."""
        mf.source_path = source_path
        mf._save_description("Hello world")
        assert (mf.source_path / "description.md").read_text() == "Hello world"


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# Tests for _save_requirements
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


class TestSaveRequirements:
    """All tests for _save_requirements."""

    def test_copies_requirements(self, mf: MakeFiles, source_path: Path) -> None:
        """Test that requirements.txt is copied to the source directory."""
        mf.source_path = source_path
        mf._save_requirements()
        result = (mf.source_path / "requirements.txt").read_text()
        assert len(result) > 0


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# Tests for _get_template
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


class TestGetTemplate:
    """All tests for _get_template."""

    def test_dataset_override_takes_priority(self, mf: MakeFiles, tmp_path: Path) -> None:
        """Test that a file in the dataset directory is preferred over templates."""
        task_path = tmp_path / "datasets" / "task_a"
        task_path.mkdir(parents=True)
        (task_path / "override.txt").write_text("# dataset override")

        result = mf._get_template("override.txt", task_path, "default")
        assert result == task_path / "override.txt"

    def test_default_fallback_returns_path(self, mf: MakeFiles, tmp_path: Path) -> None:
        """Test that a nonexistent backend falls back to the default template."""
        task_path = tmp_path / "datasets" / "task_a"
        task_path.mkdir(parents=True)

        result = mf._get_template("some_file.txt", task_path, "nonexistent_backend")
        assert result == mf.template_path / "default" / "some_file.txt"

    def test_real_fixed_files_resolve(self, mf: MakeFiles, config: dict[str, Any]) -> None:
        """Test that every fixed_file in task_spec resolves to an existing template."""
        task_ids = mf._normalize_task_ids(config, "train")
        template_backend = config.get("template_backend", "default")
        task_path = mf.base_path / "datasets" / task_ids[0]

        for fixed_file in mf.task_spec["fixed_files"]:
            result = mf._get_template(fixed_file, task_path, template_backend)
            assert result.exists(), (
                f"Template for fixed file '{fixed_file}' not found. "
                f"Checked: {task_path / fixed_file}, "
                f"{mf.template_path / template_backend / fixed_file}, "
                f"{mf.template_path / 'default' / fixed_file}"
            )

    def test_real_module_base_files_resolve(self, mf: MakeFiles, config: dict[str, Any]) -> None:
        """Test that every module_file base/ template resolves to an existing file."""
        task_ids = mf._normalize_task_ids(config, "train")
        template_backend = config.get("template_backend", "default")
        task_path = mf.base_path / "datasets" / task_ids[0]

        for module_file in mf.task_spec["module_files"]:
            result = mf._get_template(f"base/{module_file}", task_path, template_backend)
            assert result.exists(), f"Base template for module '{module_file}' not found."


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# Tests for _dir_empty
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


class TestDirEmpty:
    """All tests for _dir_empty."""

    def test_nonexistent_returns_true(self, mf: MakeFiles, tmp_path: Path) -> None:
        """Test that a nonexistent path is considered empty."""
        assert mf._dir_empty(tmp_path / "nope") is True

    def test_empty_dir_returns_true(self, mf: MakeFiles, tmp_path: Path) -> None:
        """Test that an empty directory is considered empty."""
        d = tmp_path / "empty"
        d.mkdir()
        assert mf._dir_empty(d) is True

    def test_nonempty_dir_returns_false(self, mf: MakeFiles, tmp_path: Path) -> None:
        """Test that a directory with contents is not considered empty."""
        d = tmp_path / "has_stuff"
        d.mkdir()
        (d / "file.txt").write_text("x")
        assert mf._dir_empty(d) is False


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# Tests for _get_download_dataset
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


class TestGetDownloadDataset:
    """All tests for _get_download_dataset."""

    def test_no_make_dataset_file_returns_none(self, mf: MakeFiles, tmp_path: Path) -> None:
        """Test that None is returned when make_dataset.py does not exist."""
        task_path = tmp_path / "datasets" / "t1"
        task_path.mkdir(parents=True)
        assert mf._get_download_dataset("t1", task_path) is None

    def test_make_dataset_without_function_returns_none(self, mf: MakeFiles, tmp_path: Path) -> None:
        """Test that None is returned when make_dataset.py lacks download_dataset."""
        task_path = tmp_path / "datasets" / "t1"
        task_path.mkdir(parents=True)
        (task_path / "make_dataset.py").write_text("# no download_dataset function\nx = 1\n")
        assert mf._get_download_dataset("t1", task_path) is None

    def test_make_dataset_with_function_returns_callable(self, mf: MakeFiles, tmp_path: Path) -> None:
        """Test that a callable is returned when make_dataset.py defines download_dataset."""
        task_path = tmp_path / "datasets" / "t1"
        task_path.mkdir(parents=True)
        (task_path / "make_dataset.py").write_text(
            "from pathlib import Path\ndef download_dataset(path):\n    Path(path / 'data.txt').write_text('hello')\n"
        )
        fn = mf._get_download_dataset("t1", task_path)
        assert callable(fn)

    def test_import_error_returns_none(self, mf: MakeFiles, tmp_path: Path) -> None:
        """Test that None is returned when make_dataset.py has an import error."""
        task_path = tmp_path / "datasets" / "t1"
        task_path.mkdir(parents=True)
        (task_path / "make_dataset.py").write_text("import nonexistent_module_xyz\n")
        assert mf._get_download_dataset("t1", task_path) is None


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# Tests for _ensure_dataset_cached_and_copied
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


class TestEnsureDatasetCachedAndCopied:
    """All tests for _ensure_dataset_cached_and_copied."""

    def test_no_data_flag_skips(self, mf: MakeFiles, tmp_path: Path) -> None:
        """Test that no_data=True skips dataset creation entirely."""
        task_path = tmp_path / "datasets" / "t1"
        task_path.mkdir(parents=True)
        dest = tmp_path / "dest"
        dest.mkdir()

        mf._ensure_dataset_cached_and_copied("t1", task_path, dest, no_data=True)
        assert not (dest / "data").exists()


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# Tests for _copy_dir
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


class TestCopyDir:
    """All tests for _copy_dir."""

    def test_copies_tree(self, mf: MakeFiles, tmp_path: Path) -> None:
        """Test that a directory tree is recursively copied."""
        src = tmp_path / "src"
        src.mkdir()
        (src / "a.txt").write_text("a")
        sub = src / "sub"
        sub.mkdir()
        (sub / "b.txt").write_text("b")

        dst = tmp_path / "dst"
        mf._copy_dir(src, dst)

        assert (dst / "a.txt").read_text() == "a"
        assert (dst / "sub" / "b.txt").read_text() == "b"

    def test_overwrites_existing_dest(self, mf: MakeFiles, tmp_path: Path) -> None:
        """Test that an existing destination is completely replaced."""
        src = tmp_path / "src"
        src.mkdir()
        (src / "new.txt").write_text("new")

        dst = tmp_path / "dst"
        dst.mkdir()
        (dst / "old.txt").write_text("old")

        mf._copy_dir(src, dst)
        assert (dst / "new.txt").read_text() == "new"
        assert not (dst / "old.txt").exists()


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# End-to-end: make_files for train and test
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


class TestMakeFilesEndToEnd:
    """End-to-end tests for the make_files entry point."""

    # TODO: No test uses make_files with change_*=True i think?

    def test_make_files_train(self, mf: MakeFiles, config_with_tmp: dict[str, Any]) -> None:
        """Test that a full train run completes and produces all expected outputs."""
        mf.make_files(config_with_tmp, train=True, use_base=True, no_data=True)

        sp = Path(config_with_tmp["source_path"])
        assert sp.exists()
        assert (sp / "description.md").exists()
        assert (sp / "requirements.txt").exists()
        assert (sp / "run_main.py").exists()
        assert (sp / "discovered").is_dir()

        task_ids = mf._normalize_task_ids(config_with_tmp, "train")
        model_ids = mf._normalize_model_ids(config_with_tmp, "train", task_ids)
        for task_id, model_id in zip(task_ids, model_ids, strict=False):
            dest = sp / f"{task_id}_{model_id}" if model_id else sp / task_id
            assert dest.exists(), f"Task directory '{dest}' missing after train make_files."

    def test_make_files_test_after_train(self, mf: MakeFiles, config_with_tmp: dict[str, Any]) -> None:
        """Test that a test run after train completes and preserves discovered/."""
        mf.make_files(config_with_tmp, train=True, use_base=True, no_data=True)
        mf.make_files(config_with_tmp, train=False, use_base=True, no_data=True)

        sp = Path(config_with_tmp["source_path"])
        assert sp.exists()
        assert (sp / "description.md").exists()
        assert (sp / "discovered").is_dir()

    def test_make_files_train_use_base_false(self, mf: MakeFiles, config_with_tmp: dict[str, Any]) -> None:
        """Test that train with use_base=False (edit templates) completes without error."""
        mf.make_files(config_with_tmp, train=True, use_base=False, no_data=True)

        sp = Path(config_with_tmp["source_path"])
        assert sp.exists()
        assert (sp / "description.md").exists()

    def test_make_files_idempotent_train(self, mf: MakeFiles, config_with_tmp: dict[str, Any]) -> None:
        """Test that running train twice produces an identical description."""
        mf.make_files(config_with_tmp, train=True, use_base=True, no_data=True)
        sp = Path(config_with_tmp["source_path"])
        first_desc = (sp / "description.md").read_text()

        mf.make_files(config_with_tmp, train=True, use_base=True, no_data=True)
        second_desc = (sp / "description.md").read_text()

        assert first_desc == second_desc

    def test_description_is_nonempty(self, mf: MakeFiles, config_with_tmp: dict[str, Any]) -> None:
        """Test that the generated description has substantial content."""
        mf.make_files(config_with_tmp, train=True, use_base=True, no_data=True)
        sp = Path(config_with_tmp["source_path"])
        desc = (sp / "description.md").read_text()
        assert len(desc) > 50

    def test_symlinks_resolve(self, mf: MakeFiles, config_with_tmp: dict[str, Any]) -> None:
        """Test that all symlinks in task directories resolve to existing files."""
        mf.make_files(config_with_tmp, train=True, use_base=True, no_data=True)

        sp = Path(config_with_tmp["source_path"])
        task_ids = mf._normalize_task_ids(config_with_tmp, "train")
        model_ids = mf._normalize_model_ids(config_with_tmp, "train", task_ids)

        for task_id, model_id in zip(task_ids, model_ids, strict=False):
            dest = sp / f"{task_id}_{model_id}" if model_id else sp / task_id
            if not dest.exists():
                continue
            for item in dest.iterdir():
                if item.is_symlink():
                    assert item.resolve().exists(), f"Broken symlink: {item} -> {os.readlink(item)}"
