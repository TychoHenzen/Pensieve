## ADDED Requirements

### Requirement: MNIST data binding via source field

The generator MUST accept a `data_source` config parameter. When set to `"mnist"`, `features` MUST contain 784 float values (28x28 pixel intensities, normalized to [0,1]), `label` MUST be the digit string (e.g. `"3"`), and `source` MUST name the dataset item (e.g. `"mnist-train-12345"`). When `data_source` is absent or `None`, the generator MUST produce synthetic features as it does today, with `source` remaining `None`. [REQUIRED]

#### Scenario: synthetic mode unchanged

- **GIVEN** no `data_source` in config
- **WHEN** the stream is generated
- **THEN** behavior is identical to the current generator (synthetic features, source is None)

#### Scenario: MNIST features have correct shape

- **GIVEN** `data_source="mnist"` in config
- **WHEN** an `Observe` payload is inspected
- **THEN** `features` has 784 elements and `source` is not None

### Requirement: Stream hash changes with data source

The stream hash MUST incorporate the `data_source` parameter. A synthetic stream and an MNIST-bound stream with the same seed MUST produce different hashes. [REQUIRED]

#### Scenario: different hashes

- **GIVEN** two configs differing only in `data_source` (None vs "mnist")
- **WHEN** stream hashes are computed
- **THEN** they differ
