# Contributing to PipeOptz

First off, thank you for considering contributing to PipeOptz! It's people like you that make open source software such a great community.

All types of contributions are encouraged and valued. This document provides guidelines for contributing to the project.

## Table of Contents
- [Reporting Bugs](#reporting-bugs)
- [Suggesting Enhancements](#suggesting-enhancements)
- [Development Workflow](#development-workflow)
  - [Setup](#setup)
  - [Linting](#linting)
  - [Testing](#testing)
  - [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Commit Message Style](#commit-message-style)

## Reporting Bugs

If you encounter a bug, please open an issue on the [GitHub issue tracker](https://github.com/centralelyon/pipeoptz/issues). A good bug report should include:

- A clear and descriptive title.
- A detailed description of the problem, including steps to reproduce it.
- The expected behavior and what actually happened.
- Information about your environment (e.g., Python version, OS).

## Suggesting Enhancements

If you have an idea for a new feature or an improvement to existing functionality, please open an issue to start a discussion. This helps ensure that the proposed change aligns with the project's goals.

## Development Workflow

### Setup

To get started with development, you need to set up your local environment. Please follow the instructions in the [Development Environment section of the README.md](./README.md#development-environment).

### Linting

We use [Pylint](https://www.pylint.org/) to identify and report on potential issues in the code. To run the linter, use the following command:

```bash
pylint src/pipeoptz
```

Please resolve any reported issues before submitting a pull request.

### Testing

Tests are written using the [pytest](https://docs.pytest.org/) framework. To run the full test suite, execute the following command from the root of the project:

```bash
pytest
```

Make sure that all tests pass before submitting your changes. If you are adding a new feature, please include corresponding tests.

### Documentation

The project documentation is built using [MkDocs](https://www.mkdocs.org/). For instructions on how to build and serve the documentation locally, please refer to the [Documentation section of the README.md](./README.md#documentation).

## Pull Request Process

1.  **Fork the repository** and create a new branch from `main` for your changes.
2.  **Set up your development environment** as described above.
3.  **Make your code changes.**
4.  **Ensure your code is well-formatted and tested:**
    - Run `pylint src/pipeoptz` and address any issues.
    - Run `pytest` to ensure all tests pass.
5.  **Commit your changes** using a descriptive commit message (see [Commit Message Style](#commit-message-style)).
6.  **Push your branch** to your fork.
7.  **Open a pull request** to the `main` branch of the original repository. Provide a clear description of the changes you have made.

- **`feat`**: A new feature.
- **`fix`**: A bug fix.
- **`docs`**: Documentation only changes.
- **`style`**: Changes that do not affect the meaning of the code (white-space, formatting, etc).
- **`refactor`**: A code change that neither fixes a bug nor adds a feature.
- **`test`**: Adding missing tests or correcting existing tests.
- **`chore`**: Changes to the build process or auxiliary tools.