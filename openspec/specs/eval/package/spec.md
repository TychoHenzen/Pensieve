## Purpose

Defines the top-level stream package contract: importability and schema version tracking.

## Requirements

### Requirement: Package importability

The `eval.stream` package MUST be importable without side effects or errors. [REQUIRED]

#### Scenario: clean import

- **WHEN** `import eval.stream` is executed
- **THEN** the module object is not `None`

#### Scenario: no side effects on import

- **WHEN** `import eval.stream` is executed
- **THEN** no files are created or modified

### Requirement: STREAM_SCHEMA_VERSION constant

`eval.stream` MUST expose a `STREAM_SCHEMA_VERSION` attribute that is a non-empty string. This version tracks the serialization schema so stored runs can detect incompatibility. [REQUIRED]

#### Scenario: version is a non-empty string

- **WHEN** `eval.stream.STREAM_SCHEMA_VERSION` is read
- **THEN** it is a `str` with length >= 1

#### Scenario: version is a str type

- **WHEN** `type(eval.stream.STREAM_SCHEMA_VERSION)` is checked
- **THEN** it is `str`
