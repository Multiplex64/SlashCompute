import pytest

from slashcompute.pipeline.tiny import make_tiny_dataset, make_tiny_model


@pytest.fixture(scope="session")
def tiny_model(tmp_path_factory):
    return make_tiny_model(tmp_path_factory.mktemp("tiny_model"))


@pytest.fixture(scope="session")
def tiny_dataset(tmp_path_factory):
    return make_tiny_dataset(tmp_path_factory.mktemp("data") / "train.jsonl")
