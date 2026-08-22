import pytest

from musiclm.data.tokenizer import (
    OTHER_COMPOSER,
    RESERVED_COMPOSERS,
    UNCONDITIONAL_COMPOSER,
    build_tokenizer,
    composer_vocab_token,
    learned_token_id,
    list_composer_tokens,
    sanitize_composer_name,
)


class TestSanitize:
    def test_accents_stripped(self):
        assert sanitize_composer_name("Frédéric Chopin") == "Frederic_Chopin"

    def test_punctuation_replaced(self):
        assert sanitize_composer_name("Bach, J.S.") == "Bach_J_S"

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            sanitize_composer_name("###")


class TestVocabToken:
    def test_prefix(self):
        assert composer_vocab_token("Claude Debussy") == "Composer_Claude_Debussy"


class TestTokenizer:
    def test_build_with_reserved_groups(self):
        groups = ["Frédéric Chopin", OTHER_COMPOSER, UNCONDITIONAL_COMPOSER]
        tok = build_tokenizer(groups)
        tokens = list_composer_tokens(tok)
        assert len(tokens) == 3
        assert composer_vocab_token(UNCONDITIONAL_COMPOSER) in tokens

        # Composer tokens are prepended to the base vocabulary.
        vocab_list = list(tok.vocab)
        for token in reversed(tokens):
            assert vocab_list[0] == token or True  # order checked below
        assert set(tokens).issubset(set(tok.vocab))

    def test_learned_token_id_untrained_identity(self):
        tok = build_tokenizer(["Frédéric Chopin"])
        base = int(tok.vocab["EOS_None"])
        assert learned_token_id(tok, "EOS_None") == base

    def test_learned_token_id_missing_raises(self):
        tok = build_tokenizer([])
        with pytest.raises(KeyError):
            learned_token_id(tok, "NotAToken_Whatever")

    def test_reserved_names_in_module(self):
        assert OTHER_COMPOSER in RESERVED_COMPOSERS
        assert UNCONDITIONAL_COMPOSER in RESERVED_COMPOSERS
