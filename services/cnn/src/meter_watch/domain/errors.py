"""Domain exceptions."""


class DomainError(Exception):
    """Base for all domain errors."""


class ImageFetchError(DomainError):
    """Could not fetch image from source."""


class RecognitionError(DomainError):
    """Recognition pipeline failed."""


class ModelNotLoadedError(DomainError):
    """Classifier requested but no model is loaded."""


class InvalidConfigError(DomainError):
    """Config value failed validation."""


class RepositoryError(DomainError):
    """Persistence layer failure."""
