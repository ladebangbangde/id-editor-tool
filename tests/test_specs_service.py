import pytest

from app.core.exceptions import InvalidArgumentError
from app.services.specs import get_photo_spec, list_photo_specs, supported_size_keys


def test_passport_alias_maps_to_passport_photo() -> None:
    spec = get_photo_spec('passport')
    assert spec.key == 'passport_photo'


def test_unsupported_size_key_returns_suggestions() -> None:
    with pytest.raises(InvalidArgumentError) as exc_info:
        get_photo_spec('passpor')

    assert 'did you mean: passport_photo' in exc_info.value.message
    assert exc_info.value.details['didYouMean'] == 'passport_photo'
    assert 'passport_photo' in exc_info.value.details['supportedSizeKeys']
    assert exc_info.value.details['customSizeSupported'] is False


def test_list_photo_specs_exposes_canonical_fields() -> None:
    keys = supported_size_keys()
    specs = list_photo_specs()

    assert 'one_inch' in keys
    assert 'passport_photo' in keys
    assert all('sizeKey' in item and 'aliases' in item for item in specs)
