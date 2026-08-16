import importlib
import os
import tempfile

import eval.stream as stream


# covers: eval/package::Package importability::clean import
def test_stream_package_imports():
    assert stream is not None


# covers: eval/package::Package importability::no side effects on import
def test_stream_import_no_side_effects():
    with tempfile.TemporaryDirectory() as d:
        before = set(os.listdir(d))
        old_cwd = os.getcwd()
        try:
            os.chdir(d)
            importlib.reload(stream)
        finally:
            os.chdir(old_cwd)
        after = set(os.listdir(d))
    assert after == before


# covers: eval/package::STREAM_SCHEMA_VERSION constant::version is a non-empty string
def test_stream_schema_version_is_non_empty_string():
    assert isinstance(stream.STREAM_SCHEMA_VERSION, str)
    assert len(stream.STREAM_SCHEMA_VERSION) >= 1


# covers: eval/package::STREAM_SCHEMA_VERSION constant::version is a str type
def test_stream_schema_version_is_str_type():
    assert type(stream.STREAM_SCHEMA_VERSION) is str
