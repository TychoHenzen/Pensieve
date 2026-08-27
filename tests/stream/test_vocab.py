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


# covers: eval/vocab::VOCAB pool size, uniqueness, order, and shape::vocabulary has at least 256 entries
def test_vocab_has_at_least_256_entries():
    assert len(VOCAB) >= 256


# covers: eval/vocab::VOCAB pool size, uniqueness, order, and shape::vocabulary has no duplicates
def test_vocab_no_duplicates():
    assert len(set(VOCAB)) == len(VOCAB)


# covers: eval/vocab::VOCAB pool size, uniqueness, order, and shape::vocabulary is sorted
def test_vocab_sorted_order():
    assert tuple(sorted(VOCAB)) == VOCAB


# covers: eval/vocab::VOCAB pool size, uniqueness, order, and shape::every entry is lowercase alphabetic
def test_vocab_every_entry_is_lowercase_alphabetic():
    for word in VOCAB:
        assert word.isalpha(), f"{word!r} is not alphabetic"
        assert word.islower(), f"{word!r} is not lowercase"


# covers: eval/vocab::VOCAB pool size, uniqueness, order, and shape::VOCAB is a tuple
def test_vocab_type_is_tuple():
    assert type(VOCAB) is tuple


# covers: eval/vocab::VOCAB pool size, uniqueness, order, and shape::no entry is an empty string
def test_vocab_no_empty_strings():
    assert all(len(word) > 0 for word in VOCAB)


# covers: eval/vocab::sample determinism, distinctness, and bounds checking::different seeds diverge
def test_sample_different_seeds_diverge():
    a = sample(random.Random(0), 10)
    b = sample(random.Random(1), 10)
    assert a != b


# covers: eval/vocab::sample determinism, distinctness, and bounds checking::n distinct words returned
def test_sample_n_distinct_words():
    result = sample(random.Random(0), 10)
    assert len(result) == 10
    assert len(set(result)) == 10


# covers: eval/vocab::chance_rate formula and validation::negative input rejected
def test_chance_rate_rejects_negative_input():
    with pytest.raises(ValueError):
        chance_rate(-1)
