## Purpose

Defines the Subject abstract base class, the CostCounters value object, and the cost-monotonicity invariant that all subject implementations must satisfy.

## Requirements

### Requirement: Subject is an abstract base class

`Subject` MUST NOT be directly instantiable. A subclass that implements only a subset of the required methods MUST also raise `TypeError` on construction. [REQUIRED]

#### Scenario: direct instantiation rejected

- **WHEN** `Subject()` is called
- **THEN** `TypeError` is raised

#### Scenario: partial implementation rejected

- **WHEN** a subclass implements only `observe` but not the remaining abstract methods
- **THEN** constructing that subclass raises `TypeError`

#### Scenario: full implementation accepted

- **WHEN** a subclass implements all abstract methods
- **THEN** constructing that subclass succeeds

#### Scenario: Subject is an ABC

- **WHEN** `Subject` is inspected
- **THEN** it inherits from `abc.ABC` or uses `ABCMeta`

### Requirement: CostCounters is a frozen dataclass with zero defaults

`CostCounters` MUST be frozen (attribute assignment raises `AttributeError`). Default construction MUST produce `steps=0`, `flops=0`, `wall_seconds=0.0`. [REQUIRED]

#### Scenario: frozen after construction

- **WHEN** an attribute on a `CostCounters` instance is assigned
- **THEN** `AttributeError` is raised

#### Scenario: zero defaults

- **WHEN** `CostCounters()` is constructed with no arguments
- **THEN** `steps` is 0, `flops` is 0, `wall_seconds` is 0.0

#### Scenario: steps defaults to zero

- **WHEN** `CostCounters()` is constructed with no arguments
- **THEN** `steps` is 0

#### Scenario: flops defaults to zero

- **WHEN** `CostCounters()` is constructed with no arguments
- **THEN** `flops` is 0

#### Scenario: wall_seconds defaults to zero

- **WHEN** `CostCounters()` is constructed with no arguments
- **THEN** `wall_seconds` is 0.0

### Requirement: Cost monotonicity

`Subject.cost()` MUST return a `CostCounters` whose `steps` field is monotonically non-decreasing across observe and answer calls. After at least one `answer` call, `steps` MUST have strictly increased from the initial value. [REQUIRED]

#### Scenario: steps increase after answer

- **WHEN** a subject observes one event and then answers one probe
- **THEN** `cost().steps` after the answer is strictly greater than `cost().steps` before

#### Scenario: steps non-decreasing across observe calls

- **WHEN** a subject observes multiple events
- **THEN** `cost().steps` never decreases between calls

#### Scenario: steps strictly increase after answer vs initial

- **WHEN** a subject answers at least one probe
- **THEN** `cost().steps` is strictly greater than `cost().steps` before any calls
