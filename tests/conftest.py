from __future__ import annotations

import os
from pathlib import Path

import pytest

from issueforge_server.config import EnvConfig
from issueforge_server.crypto import SecretBox
from issueforge_server.db import init_db, make_session_factory
from issueforge_server.kb.embed import HashEmbedder


@pytest.fixture()
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> EnvConfig:
    monkeypatch.setenv("ISSUEFORGE_SERVER_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("ISSUEFORGE_EMBEDDING", "hash")
    return EnvConfig.from_env()


@pytest.fixture()
def engine(env: EnvConfig):
    engine = init_db(env.paths.db_path)
    yield engine
    engine.dispose()


@pytest.fixture()
def factory(engine):
    return make_session_factory(engine)


@pytest.fixture()
def session(factory):
    session = factory()
    yield session
    session.close()


@pytest.fixture()
def box(env: EnvConfig) -> SecretBox:
    return SecretBox.from_path(env.paths.secret_path)


@pytest.fixture()
def embedder() -> HashEmbedder:
    return HashEmbedder()


@pytest.fixture()
def example_task(tmp_path: Path) -> Path:
    path = tmp_path / "task_relu.py"
    path.write_text(
        "import torch\n"
        "import torch.nn as nn\n\n"
        "class Model(nn.Module):\n"
        "    def forward(self, x):\n"
        "        return torch.relu(x)\n\n"
        "def get_inputs():\n"
        "    return [torch.randn(128, 128)]\n\n"
        "def get_init_inputs():\n"
        "    return []\n",
        encoding="utf-8",
    )
    return path
