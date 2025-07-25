import pytest
import numpy as np
from math import comb

import sys, os
sys.path.append(os.path.abspath("../"))
from pipeoptz.parameter import (
    Parameter,
    IntParameter,
    FloatParameter,
    ChoiceParameter,
    MultiChoiceParameter,
    BoolParameter
)


# --- Parameter class tests ---

class TestParameter:
    def test_base_class_not_implemented(self):
        """
        Tests that the base class methods raise NotImplementedError.
        """
        p = Parameter("node1", "param1")
        with pytest.raises(NotImplementedError):
            p.get_parametric_space()
        with pytest.raises(NotImplementedError):
            p.get_random_value()

    def test_base_class_set_get_value(self):
        """
        Tests the basic set/get functionality of the base class.
        """
        p = Parameter("node1", "param1")
        p.set_value(100)
        assert p.get_value() == 100


# --- IntParameter tests ---

class TestIntParameter:
    def test_initialization(self):
        """
        Tests the initialization of an IntParameter.
        """
        p = IntParameter("node1", "p1", 0, 10)
        assert p.node_id == "node1"
        assert p.param_name == "p1"
        assert p.min_value == 0
        assert p.max_value == 10
        assert 0 <= p.get_value() <= 10

    def test_init_invalid_types_raises_error(self):
        """
        Tests that initialization with invalid types raises an error.
        """
        with pytest.raises(ValueError, match="min_value and max_value must be integers."):
            IntParameter("node1", "p1", 0.5, 10)
        with pytest.raises(ValueError, match="min_value and max_value must be integers."):
            IntParameter("node1", "p1", 0, "10")

    def test_init_min_greater_than_max_raises_error(self):
        """
        Tests that initialization with min > max raises an error.
        """
        with pytest.raises(ValueError, match="min_value cannot be greater than max_value."):
            IntParameter("node1", "p1", 11, 10)

    def test_set_value_valid(self):
        """
        Tests the set_value method of an IntParameter.
        """
        p = IntParameter("node1", "p1", 0, 10)
        p.set_value(5)
        assert p.get_value() == 5

    def test_set_value_invalid_type_raises_error(self):
        """
        Tests that setting an invalid value type raises an error.
        """
        p = IntParameter("node1", "p1", 0, 10)
        with pytest.raises(ValueError, match="Value must be an integer"):
            p.set_value(5.5)

    def test_set_value_out_of_range_raises_error(self):
        """
        Tests that setting a value out of range raises an error.
        """
        p = IntParameter("node1", "p1", 0, 10)
        with pytest.raises(ValueError, match="Value must be between"):
            p.set_value(11)
        with pytest.raises(ValueError, match="Value must be between"):
            p.set_value(-1)

    def test_get_random_value(self):
        """
        Tests the get_random_value method of an IntParameter.
        """
        p = IntParameter("node1", "p1", 5, 5)
        val = p.get_random_value()
        assert val == 5

    def test_get_random_value_sets_value(self):
        """
        Tests that get_random_value sets the value if set_value is True.
        """
        p = IntParameter("node1", "p1", 0, 10000)
        old_val = p.get_value()
        rand_val = p.get_random_value(set_value=True)
        assert p.get_value() == rand_val
        assert old_val != rand_val  # High probability

    def test_get_parametric_space(self):
        """
        Tests the get_parametric_space method of an IntParameter.
        """
        p = IntParameter("node1", "p1", 0, 10)
        space = p.get_parametric_space()
        assert space == {"type": int, "dim": 1, "range_size": 10}


# --- FloatParameter tests ---

class TestFloatParameter:
    def test_initialization(self):
        """
        Tests the initialization of a FloatParameter.
        """
        p = FloatParameter("node1", "p1", 0.0, 10.0)
        assert p.min_value == 0.0
        assert p.max_value == 10.0
        assert 0.0 <= p.get_value() <= 10.0

    def test_init_with_int(self):
        """
        Tests the initialization of a FloatParameter with integer bounds.
        """
        p = FloatParameter("node1", "p1", 0, 10)
        assert isinstance(p.min_value, float)
        assert isinstance(p.max_value, float)

    def test_set_value_valid(self):
        """
        Tests the set_value method of a FloatParameter.
        """
        p = FloatParameter("node1", "p1", 0.0, 10.0)
        p.set_value(5.5)
        assert p.get_value() == 5.5
        p.set_value(5)  # Should cast int to float
        assert p.get_value() == 5.0

    def test_set_value_out_of_range_raises_error(self):
        """
        Tests that setting a value out of range raises an error.
        """
        p = FloatParameter("node1", "p1", 0.0, 10.0)
        with pytest.raises(ValueError, match="Value must be between"):
            p.set_value(10.1)

    def test_get_parametric_space(self):
        p = FloatParameter("node1", "p1", 0.0, 10.0)
        space = p.get_parametric_space()
        assert space == {"type": float, "range_size": 10.0}


# --- ChoiceParameter tests ---

class TestChoiceParameter:
    def test_initialization(self):
        """
        Tests the initialization of a ChoiceParameter.
        """
        choices = ["a", "b", "c"]
        p = ChoiceParameter("node1", "p1", choices)
        assert p.choices == choices
        assert p.get_value() in choices

    def test_set_value_valid(self):
        """
        Tests the set_value method of a ChoiceParameter.
        """
        p = ChoiceParameter("node1", "p1", ["a", "b", "c"])
        p.set_value("b")
        assert p.get_value() == "b"

    def test_set_value_invalid_raises_error(self):
        """
        Tests that setting an invalid value raises an error.
        """
        p = ChoiceParameter("node1", "p1", ["a", "b", "c"])
        with pytest.raises(ValueError, match="Value must be one of the following options"):
            p.set_value("d")

    def test_get_random_value(self):
        """
        Tests the get_random_value method of a ChoiceParameter.
        """
        choices = ["a", "b", "c"]
        p = ChoiceParameter("node1", "p1", choices)
        rand_val = p.get_random_value()
        assert rand_val in choices

    def test_get_parametric_space(self):
        """
        Tests the get_parametric_space method of a ChoiceParameter.
        """
        p = ChoiceParameter("node1", "p1", ["a", "b", "c"])
        space = p.get_parametric_space()
        assert space == {"type": None, "range_size": 3}


# --- MultiChoiceParameter tests ---

class TestMultiChoiceParameter:
    def test_initialization_fixed_size(self):
        """
        Tests the initialization of a MultiChoiceParameter with a fixed size.
        """
        choices = ["a", "b", "c", "d"]
        p = MultiChoiceParameter("node1", "p1", choices, min_choices=2, max_choices=None)
        assert p.min_choices == 2
        assert p.max_choices == 2
        assert len(p.get_value()) == 2

    def test_initialization_range_size(self):
        """
        Tests the initialization of a MultiChoiceParameter with a range size.
        """
        choices = ["a", "b", "c", "d"]
        p = MultiChoiceParameter("node1", "p1", choices, min_choices=1, max_choices=3)
        assert p.min_choices == 1
        assert p.max_choices == 3
        assert 1 <= len(p.get_value()) <= 3

    def test_initialization_max_zero(self):
        """
        Tests the initialization of a MultiChoiceParameter with max_choices=0.
        """
        choices = ["a", "b", "c", "d"]
        p = MultiChoiceParameter("node1", "p1", choices, min_choices=1, max_choices=0)
        assert p.max_choices == 4

    def test_init_invalid_constraints_raises_error(self):
        """
        Tests that initialization with invalid constraints raises an error.
        """
        with pytest.raises(ValueError, match="min_choices cannot be greater than max_choices"):
            MultiChoiceParameter("node1", "p1", ["a"], min_choices=2, max_choices=1)
        with pytest.raises(ValueError, match="min_choices must be at least 1"):
            MultiChoiceParameter("node1", "p1", ["a"], min_choices=0)

    def test_set_value_valid(self):
        """
        Tests the set_value method of a MultiChoiceParameter.
        """
        p = MultiChoiceParameter("node1", "p1", ["a", "b", "c"], min_choices=1, max_choices=2)
        p.set_value(["a", "c"])
        assert p.get_value() == ["a", "c"]

    def test_set_value_invalid_type_raises_error(self):
        """
        Tests that setting an invalid value type raises an error.
        """
        p = MultiChoiceParameter("node1", "p1", ["a", "b", "c"])
        with pytest.raises(ValueError, match="Value must be a list"):
            p.set_value("a")

    def test_set_value_invalid_size_raises_error(self):
        """
        Tests that setting an invalid value size raises an error.
        """
        p = MultiChoiceParameter("node1", "p1", ["a", "b", "c"], min_choices=2, max_choices=2)
        with pytest.raises(ValueError, match="must contain between 2 and 2 items"):
            p.set_value(["a"])

    def test_set_value_invalid_item_raises_error(self):
        """
        Tests that setting an invalid value item raises an error.
        """
        p = MultiChoiceParameter("node1", "p1", ["a", "b", "c"])
        with pytest.raises(ValueError, match="is not in the allowed choices"):
            p.set_value(["d"])

    def test_get_random_value(self):
        """
        Tests the get_random_value method of a MultiChoiceParameter.
        """
        choices = ["a", "b", "c", "d", "e"]
        p = MultiChoiceParameter("node1", "p1", choices, min_choices=2, max_choices=4)
        rand_val = p.get_random_value()
        assert isinstance(rand_val, list)
        assert 2 <= len(rand_val) <= 4
        assert all(item in choices for item in rand_val)

    def test_get_parametric_space(self):
        """
        Tests the get_parametric_space method of a MultiChoiceParameter.
        """
        choices = ["a", "b", "c", "d"]  # n=4
        p = MultiChoiceParameter("node1", "p1", choices, min_choices=2, max_choices=3)
        # C(4,2) + C(4,3) = 6 + 4 = 10
        expected_size = comb(4, 2) + comb(4, 3)
        space = p.get_parametric_space()
        assert space == {"type": None, "range_size": expected_size}


# --- BoolParameter tests ---

class TestBoolParameter:
    def test_initialization(self):
        """
        Tests the initialization of a BoolParameter.
        """
        p = BoolParameter("node1", "p1")
        assert p.get_value() in [True, False]

    def test_set_value_valid(self):
        """        
        Tests the set_value method of a BoolParameter.
        """
        p = BoolParameter("node1", "p1")
        p.set_value(True)
        assert p.get_value() is True
        p.set_value(False)
        assert p.get_value() is False

    def test_set_value_invalid_type_raises_error(self):
        """
        Tests that setting an invalid value type raises an error.
        """
        p = BoolParameter("node1", "p1")
        with pytest.raises(ValueError, match="Value must be a boolean"):
            p.set_value(1)

    def test_get_random_value(self):
        """
        Tests the get_random_value method of a BoolParameter.
        """
        p = BoolParameter("node1", "p1")
        rand_val = p.get_random_value()
        assert isinstance(rand_val, bool)

    def test_get_parametric_space(self):
        """
        Tests the get_parametric_space method of a BoolParameter.
        """
        p = BoolParameter("node1", "p1")
        space = p.get_parametric_space()
        assert space == {"type": bool, "range_size": 2}
