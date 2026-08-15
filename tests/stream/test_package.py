import eval.stream as stream


# covers: eval/package::Package importability::clean import
def test_stream_package_imports():
    assert stream is not None


# covers: eval/package::STREAM_SCHEMA_VERSION constant::version is a non-empty string
def test_stream_schema_version_is_non_empty_string():
    assert isinstance(stream.STREAM_SCHEMA_VERSION, str)
    assert len(stream.STREAM_SCHEMA_VERSION) >= 1
