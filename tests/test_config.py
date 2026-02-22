"""Tests for crispen.config — 100% branch coverage."""

from crispen.config import CrispenConfig, _apply, _read_toml, load_config


# ---------------------------------------------------------------------------
# _read_toml
# ---------------------------------------------------------------------------


def test_read_toml_success(tmp_path):
    toml_file = tmp_path / "test.toml"
    toml_file.write_text("[tool.crispen]\nmin_tuple_size = 5\n", encoding="utf-8")
    result = _read_toml(toml_file)
    assert result == {"tool": {"crispen": {"min_tuple_size": 5}}}


def test_read_toml_missing_file(tmp_path):
    result = _read_toml(tmp_path / "nonexistent.toml")
    assert result == {}


def test_read_toml_invalid_toml(tmp_path):
    bad_file = tmp_path / "bad.toml"
    bad_file.write_bytes(b"\x80\x81\x82")  # invalid UTF-8 → parse error
    result = _read_toml(bad_file)
    assert result == {}


# ---------------------------------------------------------------------------
# _apply
# ---------------------------------------------------------------------------


def test_apply_empty_dict():
    cfg = CrispenConfig()
    _apply(cfg, {})
    assert cfg == CrispenConfig()  # unchanged


def test_apply_known_key():
    cfg = CrispenConfig()
    _apply(cfg, {"min_tuple_size": 6})
    assert cfg.min_tuple_size == 6


def test_apply_unknown_key_ignored():
    cfg = CrispenConfig()
    _apply(cfg, {"unknown_option": 999})
    assert cfg == CrispenConfig()  # unchanged


def test_apply_mixed_keys():
    cfg = CrispenConfig()
    _apply(cfg, {"min_tuple_size": 7, "bogus": "x"})
    assert cfg.min_tuple_size == 7


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------


def test_load_config_defaults_when_no_files(tmp_path):
    cfg = load_config(project_root=tmp_path)
    assert cfg == CrispenConfig()


def test_load_config_reads_pyproject_toml(tmp_path):
    (tmp_path / "pyproject.toml").write_text(
        "[tool.crispen]\nmin_tuple_size = 6\n", encoding="utf-8"
    )
    cfg = load_config(project_root=tmp_path)
    assert cfg.min_tuple_size == 6
    # Other fields stay at defaults
    assert cfg.min_duplicate_weight == 3


def test_load_config_pyproject_without_crispen_section(tmp_path):
    (tmp_path / "pyproject.toml").write_text(
        "[tool.pytest]\naddopts = '--cov'\n", encoding="utf-8"
    )
    cfg = load_config(project_root=tmp_path)
    assert cfg == CrispenConfig()


def test_load_config_local_overrides_pyproject(tmp_path):
    (tmp_path / "pyproject.toml").write_text(
        "[tool.crispen]\nmin_tuple_size = 5\nmodel = 'old-model'\n",
        encoding="utf-8",
    )
    (tmp_path / ".crispen.toml").write_text("model = 'new-model'\n", encoding="utf-8")
    cfg = load_config(project_root=tmp_path)
    assert cfg.min_tuple_size == 5  # from pyproject
    assert cfg.model == "new-model"  # overridden by .crispen.toml


def test_load_config_uses_cwd_when_no_root(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").write_text(
        "[tool.crispen]\nmax_duplicate_seq_len = 12\n", encoding="utf-8"
    )
    cfg = load_config()  # no project_root → uses cwd
    assert cfg.max_duplicate_seq_len == 12


def test_load_config_all_options(tmp_path):
    (tmp_path / "pyproject.toml").write_text(
        "[tool.crispen]\n"
        "min_duplicate_weight = 5\n"
        "max_duplicate_seq_len = 10\n"
        "min_tuple_size = 6\n"
        "model = 'claude-opus-4-6'\n"
        "max_function_length = 100\n"
        "helper_docstrings = true\n"
        "update_diff_file_callers = false\n",
        encoding="utf-8",
    )
    cfg = load_config(project_root=tmp_path)
    assert cfg.min_duplicate_weight == 5
    assert cfg.max_duplicate_seq_len == 10
    assert cfg.min_tuple_size == 6
    assert cfg.model == "claude-opus-4-6"
    assert cfg.max_function_length == 100
    assert cfg.helper_docstrings is True
    assert cfg.update_diff_file_callers is False
