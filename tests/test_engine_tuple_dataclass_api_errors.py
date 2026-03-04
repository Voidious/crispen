from unittest.mock import patch
import pytest
from crispen.engine import _apply_tuple_dataclass
from crispen.errors import CrispenAPIError


def test_apply_tuple_dataclass_crispen_api_error():
    with patch("crispen.engine.MetadataWrapper") as MockWrapper:
        MockWrapper.return_value.visit.side_effect = CrispenAPIError("test api error")
        with pytest.raises(CrispenAPIError):
            _apply_tuple_dataclass("f.py", [(1, 1)], "x = 1\n", False, set())
