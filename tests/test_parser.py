"""Tests for RoleFillerParser and EmbeddingResolver (Stage 20)."""

import torch
import pytest

from snks.dcam.hac import HACEngine
from snks.language.chunker import Chunk
from snks.language.parser import EmbeddingResolver, RoleFillerParser
from snks.language.roles import get_roles


@pytest.fixture
def hac():
    return HACEngine(dim=2048)


@pytest.fixture
def roles():
    return get_roles(hac_dim=2048)


@pytest.fixture
def parser(hac, roles):
    return RoleFillerParser(hac, roles)


def _random_embedding(seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    v = torch.randn(2048, generator=gen)
    return v / v.norm()


class TestRoleFillerParser:

    def test_parse_returns_unit_vector(self, parser):
        chunks = [Chunk("cat", "AGENT"), Chunk("sits", "ACTION")]
        emb = {"cat": _random_embedding(1), "sits": _random_embedding(2)}
        result = parser.parse(chunks, emb)
        assert result.shape == (2048,)
        assert abs(result.norm().item() - 1.0) < 1e-4

    def test_extract_recovers_filler(self, parser, hac):
        cat_emb = _random_embedding(1)
        sits_emb = _random_embedding(2)
        mat_emb = _random_embedding(3)

        chunks = [
            Chunk("cat", "AGENT"),
            Chunk("sits", "ACTION"),
            Chunk("mat", "LOCATION"),
        ]
        emb = {"cat": cat_emb, "sits": sits_emb, "mat": mat_emb}

        sentence_hac = parser.parse(chunks, emb)

        recovered_cat = parser.extract("AGENT", sentence_hac)
        recovered_sits = parser.extract("ACTION", sentence_hac)
        recovered_mat = parser.extract("LOCATION", sentence_hac)

        # Correct extractions should have positive similarity.
        # With 3 bindings in bundle, power-spectrum unbind yields ~0.15-0.25.
        # Random noise is ~0.02, so 0.1 is a safe threshold.
        assert hac.similarity(recovered_cat, cat_emb) > 0.1
        assert hac.similarity(recovered_sits, sits_emb) > 0.1
        assert hac.similarity(recovered_mat, mat_emb) > 0.1

    def test_wrong_role_low_similarity(self, parser, hac):
        cat_emb = _random_embedding(1)
        sits_emb = _random_embedding(2)

        chunks = [Chunk("cat", "AGENT"), Chunk("sits", "ACTION")]
        emb = {"cat": cat_emb, "sits": sits_emb}

        sentence_hac = parser.parse(chunks, emb)

        # Extracting AGENT should NOT match "sits" embedding.
        recovered = parser.extract("AGENT", sentence_hac)
        sim_wrong = hac.similarity(recovered, sits_emb)
        sim_right = hac.similarity(recovered, cat_emb)
        assert sim_right > sim_wrong + 0.2

    def test_extract_all(self, parser, hac):
        cat_emb = _random_embedding(1)
        sits_emb = _random_embedding(2)

        chunks = [Chunk("cat", "AGENT"), Chunk("sits", "ACTION")]
        emb = {"cat": cat_emb, "sits": sits_emb}

        sentence_hac = parser.parse(chunks, emb)
        all_extracted = parser.extract_all(sentence_hac)

        assert "AGENT" in all_extracted
        assert "ACTION" in all_extracted
        assert hac.similarity(all_extracted["AGENT"], cat_emb) > 0.1

    def test_four_roles(self, parser, hac):
        """Test with 4 roles: ATTR + AGENT + ACTION + LOCATION."""
        embs = {
            "red": _random_embedding(10),
            "cat": _random_embedding(11),
            "sits": _random_embedding(12),
            "mat": _random_embedding(13),
        }
        chunks = [
            Chunk("red", "ATTR"),
            Chunk("cat", "AGENT"),
            Chunk("sits", "ACTION"),
            Chunk("mat", "LOCATION"),
        ]
        sentence_hac = parser.parse(chunks, embs)

        for chunk in chunks:
            recovered = parser.extract(chunk.role, sentence_hac)
            sim = hac.similarity(recovered, embs[chunk.text])
            assert sim > 0.1, f"role={chunk.role} sim={sim:.3f}"

    def test_compositional_generalization(self, parser, hac):
        """Unbind works on novel combinations of known words."""
        cat_emb = _random_embedding(1)
        runs_emb = _random_embedding(4)
        floor_emb = _random_embedding(5)

        # Never seen "cat runs on floor" before — only individual words.
        chunks = [
            Chunk("cat", "AGENT"),
            Chunk("runs", "ACTION"),
            Chunk("floor", "LOCATION"),
        ]
        emb = {"cat": cat_emb, "runs": runs_emb, "floor": floor_emb}

        sentence_hac = parser.parse(chunks, emb)
        recovered = parser.extract("AGENT", sentence_hac)
        assert hac.similarity(recovered, cat_emb) > 0.1


class TestEmbeddingResolver:

    def test_resolve_known_word(self):
        class FakeGroundingMap:
            def word_to_sks(self, w):
                return 42 if w == "cat" else None

        emb_42 = _random_embedding(42)
        resolver = EmbeddingResolver(FakeGroundingMap(), embedder=None)
        result = resolver.resolve("cat", sks_embeddings={42: emb_42})
        assert torch.allclose(result, emb_42)

    def test_resolve_unknown_returns_none(self):
        class FakeGroundingMap:
            def word_to_sks(self, w):
                return None

        resolver = EmbeddingResolver(FakeGroundingMap(), embedder=None)
        assert resolver.resolve("unknown") is None

    def test_cache_embeddings(self):
        class FakeGroundingMap:
            def word_to_sks(self, w):
                return 42 if w == "cat" else None

        emb_42 = _random_embedding(42)
        resolver = EmbeddingResolver(FakeGroundingMap(), embedder=None)
        resolver.cache_embeddings({42: emb_42})
        result = resolver.resolve("cat")
        assert torch.allclose(result, emb_42)
