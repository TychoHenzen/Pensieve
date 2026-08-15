import random

import pytest

from eval.stream.vocab import VOCAB, chance_rate, sample


# covers: eval/vocab::VOCAB pool size, uniqueness, order, and shape::vocabulary size and shape
def test_vocab_is_a_tuple():
    assert isinstance(VOCAB, tuple)


# covers: eval/vocab::VOCAB_VERSION tracks vocabulary changes::vocabulary content changes
def test_vocab_version_is_a_non_empty_string():
    from eval.stream.vocab import VOCAB_VERSION
    assert isinstance(VOCAB_VERSION, str)
    assert len(VOCAB_VERSION) >= 1


# covers: eval/vocab::VOCAB pool size, uniqueness, order, and shape::vocabulary size and shape
def test_vocab_has_no_duplicates():
    assert len(VOCAB) == len(set(VOCAB))


# covers: eval/vocab::VOCAB pool size, uniqueness, order, and shape::vocabulary size and shape
def test_vocab_has_at_least_256_words():
    assert len(VOCAB) >= 256


# covers: eval/vocab::VOCAB pool size, uniqueness, order, and shape::vocabulary size and shape
def test_vocab_is_sorted():
    assert list(VOCAB) == sorted(VOCAB)


# covers: eval/vocab::VOCAB pool size, uniqueness, order, and shape::vocabulary size and shape
def test_vocab_words_are_short_lowercase():
    assert all(word == word.lower() and word.isalpha() for word in VOCAB)


# covers: eval/vocab::sample determinism, distinctness, and bounds checking::reproducible draw
def test_sample_is_reproducible_from_the_same_seed():
    first = sample(random.Random(0), 10)
    second = sample(random.Random(0), 10)
    assert first == second


# covers: eval/vocab::sample determinism, distinctness, and bounds checking::reproducible draw
def test_sample_diverges_across_seeds():
    first = sample(random.Random(0), 10)
    second = sample(random.Random(1), 10)
    assert first != second


# covers: eval/vocab::sample determinism, distinctness, and bounds checking::reproducible draw
def test_sample_returns_distinct_words():
    drawn = sample(random.Random(0), 20)
    assert len(drawn) == len(set(drawn))


# covers: eval/vocab::sample determinism, distinctness, and bounds checking::oversized draw rejected
def test_sample_larger_than_vocabulary_raises():
    with pytest.raises(ValueError):
        sample(random.Random(0), len(VOCAB) + 1)


# covers: eval/vocab::sample determinism, distinctness, and bounds checking::negative draw rejected
def test_sample_negative_n_raises():
    with pytest.raises(ValueError):
        sample(random.Random(0), -1)


# covers: eval/vocab::chance_rate formula and validation::chance rate for the full vocabulary
def test_chance_rate_for_full_vocabulary_is_one_over_length():
    assert chance_rate(len(VOCAB)) == pytest.approx(1 / len(VOCAB))


# covers: eval/vocab::chance_rate formula and validation::chance rate for the full vocabulary
def test_chance_rate_for_smaller_choice_set():
    assert chance_rate(4) == pytest.approx(0.25)


# covers: eval/vocab::chance_rate formula and validation::non-positive input rejected
def test_chance_rate_rejects_non_positive_choices():
    with pytest.raises(ValueError):
        chance_rate(0)
