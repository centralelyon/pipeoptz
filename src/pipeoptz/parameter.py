import numpy as np
import random as rd
from math import comb
from typing import Any, Union, List, Dict

class Parameter:
    """
    Base class for parameters used in pipeline nodes.

    This abstract class defines the interface for all parameter types.
    It should not be instantiated directly. Subclasses must implement
    methods for setting, getting, and generating random values, as well
    as describing their parametric space.

    Attributes:
        node_id (str): The ID of the node this parameter belongs to.
        param_name (str): The name of the parameter.
        value: The current value of the parameter.
    """
    def __init__(self, node_id: str, param_name: str) -> None:
        """Initializes the base parameter."""
        self.node_id = node_id
        self.param_name = param_name
        self.value: Any = None

    def set_value(self, value: Any) -> None:
        """
        Sets the value of the parameter.
        This should be overridden in subclasses.
        """
        self.value = value
    
    def get_value(self) -> Any:
        """
        Returns the current value of the parameter.
        This should be overridden in subclasses.
        """
        return self.value

    def get_random_value(self, set_value: bool = False) -> Any:
        """
        Returns a random value for the parameter.
        This should be overridden in subclasses.
        """
        raise NotImplementedError("This method must be implemented in subclasses.")

    def get_description(self) -> Dict[str, Any]:
        raise NotImplementedError("This method must be implemented in subclasses.")


class IntParameter(Parameter):
    """
    Represents an integer parameter with a defined range.

    This parameter type is used for node inputs that must be an integer
    between a specified minimum and maximum value.

    Attributes:
        node_id (str): The ID of the node this parameter belongs to.
        param_name (str): The name of the parameter.
        min_value (int): The minimum allowed value for the parameter.
        max_value (int): The maximum allowed value for the parameter.
        value (int): The current value of the parameter.
    """
    MAXINT: int = 2**63 - 1
    MININT: int = -2**63

    def __init__(self, node_id: str, param_name: str, min_value: int, max_value: int, step: int = 1) -> None:
        """
        Initializes an IntParameter.

        Args:
            node_id (str): The ID of the node this parameter belongs to.
            param_name (str): The name of the parameter.
            min_value (int): The minimum allowed value.
            max_value (int): The maximum allowed value.

        Raises:
            ValueError: If min_value or max_value are not integers, or if
                        min_value is greater than max_value.
        """
        if not isinstance(min_value, (int, np.integer)) or not isinstance(max_value, (int, np.integer)):
            raise ValueError("min_value and max_value must be integers.")
        if min_value > max_value:
            raise ValueError("min_value cannot be greater than max_value.")
        super(IntParameter, self).__init__(node_id, param_name)
        self.min_value: int = max(min_value, self.MININT)
        self.max_value: int = min(max_value, self.MAXINT)
        self.step: int = step
        self.get_random_value(True)
    
    def set_value(self, value: int) -> None:
        """
        Sets the value of the parameter, ensuring it is within the valid range.

        Args:
            value (int): The integer value to set.

        Raises:
            ValueError: If the value is not an integer or is outside the defined range.
        """
        if not isinstance(value, (int, np.integer)):
            raise ValueError(f"Value must be an integer, but 'value' is of type {type(value)}.")
        elif value < self.min_value or value > self.max_value:
            raise ValueError(f"Value must be between {self.min_value} and {self.max_value}.")
        elif (value-self.min_value) % self.step:
            raise ValueError(f"Value must be in range({self.min_value}, {self.max_value+1}, {self.step}).")
        self.value = value

    def get_value(self) -> int:
        """Returns the current value of the parameter."""
        return self.value
    
    def get_random_value(self, set_value: bool = False) -> int:
        """
        Generates a random integer within the defined range.

        Args:
            set_value (bool, optional): If True, sets the parameter's current value
                to the new random value. Defaults to False.

        Returns:
            int: A random integer.
        """
        r: int = rd.randrange(self.min_value, self.max_value+1, self.step)
        if set_value:
            self.set_value(r)
        return r

    def get_description(self) -> Dict[str, Any]:
        return {"type": "int", "value": self.value, "min": self.min_value, "max": self.max_value, "step": self.step}


class FloatParameter(Parameter):
    """
    Represents a floating-point parameter with a defined range.

    This parameter type is used for node inputs that must be a float
    between a specified minimum and maximum value.

    Attributes:
        node_id (str): The ID of the node this parameter belongs to.
        param_name (str): The name of the parameter.
        min_value (float): The minimum allowed value for the parameter.
        max_value (float): The maximum allowed value for the parameter.
        value (float): The current value of the parameter.
    """
    MAXFLOAT: float = 1.7976931348623157e+308
    MINFLOAT: float = -1.7976931348623157e+308

    def __init__(self, node_id: str, param_name: str, min_value: float, max_value: float, step: float = 0) -> None:
        """
        Initializes a FloatParameter.

        Args:
            node_id (str): The ID of the node this parameter belongs to.
            param_name (str): The name of the parameter.
            min_value (float): The minimum allowed value.
            max_value (float): The maximum allowed value.

        Raises:
            ValueError: If min_value or max_value are not floats, or if
                        min_value is greater than max_value.
        """
        if not isinstance(min_value, (float, int, np.integer, np.floating)) or not isinstance(max_value, (float, int, np.integer, np.floating)):
            raise ValueError("min_value and max_value must be floats.")
        if min_value > max_value:
            raise ValueError("min_value cannot be greater than max_value.")
        
        super(FloatParameter, self).__init__(node_id, param_name)
        self.min_value: float = max(float(min_value), self.MINFLOAT)
        self.max_value: float = min(float(max_value), self.MAXFLOAT)
        self.step: float = step
        self.get_random_value(True)

    def set_value(self, value: float) -> None:
        """
        Sets the value of the parameter, ensuring it is within the valid range.

        Args:
            value (float): The float value to set.

        Raises:
            ValueError: If the value is not a float or is outside the defined range.
        """
        if not isinstance(value, (float, int)):
            raise ValueError(f"The value must be a float, but 'value' is of type {type(value)}.")
        elif value < self.min_value or value > self.max_value:
            raise ValueError(f"Value must be between {self.min_value} and {self.max_value}.")
        elif self.step and (value-self.min_value) % self.step > self.step/10:
            raise ValueError(f"Value must be in range({self.min_value}, {self.max_value} included, {self.step}).")
        
        if self.step:
            self.value = float(value-(value-self.min_value)%self.step)
        else:
            self.value = float(value)
    
    def get_value(self) -> float:
        """
        Returns the current value of the parameter."""
        return self.value

    def get_random_value(self, set_value: bool = False) -> float:
        """
        Generates a random float within the defined range.

        Args:
            set_value (bool, optional): If True, sets the parameter's current value
                to the new random value. Defaults to False.

        Returns:
            float: A random float.
        """
        r: float = rd.uniform(self.min_value, self.max_value)
        if self.step:
            r -= (r-self.min_value)%self.step
        if set_value:
            self.set_value(r)
        return r

    def get_description(self) -> Dict[str, Any]:
        return {"type": "float", "value": self.value, "min": self.min_value, "max": self.max_value, "step": self.step}

class ChoiceParameter(Parameter):
    """
    Represents a parameter that must be chosen from a predefined list of options.

    This is used for categorical parameters where the input must be one of the
    given choices.

    Attributes:
        node_id (str): The ID of the node this parameter belongs to.
        param_name (str): The name of the parameter.
        choices (list): The list of valid options for this parameter.
        value: The currently selected choice.
    """
    def __init__(self, node_id: str, param_name: str, choices: List[Any]) -> None:
        """
        Initializes a ChoiceParameter.

        Args:
            node_id (str): The ID of the node this parameter belongs to.
            param_name (str): The name of the parameter.
            choices (list): A list of the possible values for the parameter.
        """
        super(ChoiceParameter, self).__init__(node_id, param_name)
        self.choices: List[Any] = choices
        self.get_random_value(True)
      
    def set_value(self, value: Any) -> None:
        """
        Sets the value of the parameter, ensuring it is a valid choice.

        Args:
            value: The value to set.

        Raises:
            ValueError: If the value is not in the list of allowed choices.
        """
        if value not in self.choices:
            raise ValueError(f"Value must be one of the following options: {self.choices}.")
        self.value = value

    def get_value(self) -> Any:
        """Returns the current value of the parameter."""
        return self.value
    
    def get_random_value(self, set_value: bool = False) -> Any:
        """
        Generates a random value from the list of choices.

        Args:
            set_value (bool, optional): If True, sets the parameter's current value
                to the new random value. Defaults to False.

        Returns:
            A random value from the choices.
        """
        r: Any = rd.choice(self.choices)
        if set_value:
            self.set_value(r)
        return r

    def get_description(self) -> Dict[str, Any]:
        return {"type": "choice", "value": self.value, "choices": self.choices}

class MultiChoiceParameter(Parameter):
    """
    Represents a parameter where multiple options can be chosen from a list.

    This allows for selecting a sub-list of items, with constraints on the
    minimum and maximum number of selections.

    Attributes:
        node_id (str): The ID of the node this parameter belongs to.
        param_name (str): The name of the parameter.
        choices (list): The list of all possible options.
        min_choices (int): The minimum number of items to select.
        max_choices (int): The maximum number of items to select.
        value (list): The currently selected list of choices.
    """
    def __init__(self, node_id: str, param_name: str, choices: List[Any], min_choices: int = 1, max_choices: Union[int, None] = None) -> None:
        """
        Initializes a MultiChoiceParameter.

        Args:
            node_id (str): The ID of the node this parameter belongs to.
            param_name (str): The name of the parameter.
            choices (list): A list of all possible values.
            min_choices (int, optional): The minimum number of choices to select.
                Defaults to 1.
            max_choices (int or None, optional): The maximum number of choices.
                - If None, the number of choices is fixed to `min_choices`.
                - If 0, there is no upper limit (up to all choices).
                - Otherwise, it's the specific upper limit.
                Defaults to None.

        Raises:
            ValueError: If constraints are invalid (e.g., min > max).
        """
        if max_choices is not None and max_choices != 0 and min_choices > max_choices:
            raise ValueError("min_choices cannot be greater than max_choices.")
        if min_choices < 1:
            raise ValueError("min_choices must be at least 1.")
        super(MultiChoiceParameter, self).__init__(node_id, param_name)
        self.choices: List[Any] = choices
        self.min_choices: int = min_choices
        if max_choices is None:
            self.max_choices: int = self.min_choices
        elif max_choices == 0:
            self.max_choices: int = len(choices)
        else:
            self.max_choices: int = max_choices

        n: int = len(self.choices)
        self._C: List[int] =  [comb(n, k) for k in range(self.min_choices, self.max_choices + 1)]
        self.get_random_value(True)

    def set_value(self, value: List[Any]) -> None:
        """
        Sets the value of the parameter, ensuring it is a valid sub-list of choices.

        Args:
            value (list): The list of selected values.

        Raises:
            ValueError: If the value is not a list, violates size constraints,
                        or contains items not in the allowed choices.
        """
        if not isinstance(value, list):
            raise ValueError(f"Value must be a list, but received type {type(value)}.")
        if len(value) < self.min_choices or len(value) > self.max_choices:
            raise ValueError(f"The list of values must contain between {self.min_choices} and {self.max_choices} items.")
        for v in value:
            if v not in self.choices:
                raise ValueError(f"The value '{v}' is not in the allowed choices: {self.choices}.")
        self.value = value

    def get_value(self) -> List[Any]:
        """Returns the current list of selected values."""
        return self.value
    
    def get_random_value(self, set_value: bool = False) -> List[Any]:
        """
        Generates a random sub-list of choices that respects the size constraints.

        Args:
            set_value (bool, optional): If True, sets the parameter's current value
                to the new random value. Defaults to False.

        Returns:
            list: A random list of choices.
        """
        k: int = rd.choices(range(self.min_choices, self.max_choices + 1), weights=self._C, k=1)[0]
        r: List[Any] = rd.sample(self.choices, k=k)
        if set_value:
            self.set_value(r)
        return r

    def get_description(self) -> Dict[str, Any]:
        return {"type": "multichoice", "value": self.value, "choices": self.choices, "min_choices": self.min_choices, "max_choices": self.max_choices}

class BoolParameter(Parameter):
    """
    Represents a boolean parameter (True or False).

    Attributes:
        node_id (str): The ID of the node this parameter belongs to.
        param_name (str): The name of the parameter.
        value (bool): The current value of the parameter.
    """
    def __init__(self, node_id: str, param_name: str) -> None:
        """
        Initializes a BoolParameter.

        Args:
            node_id (str): The ID of the node this parameter belongs to.
            param_name (str): The name of the parameter.
        """
        super(BoolParameter, self).__init__(node_id, param_name)
        self.get_random_value(True)

    def set_value(self, value: bool) -> None:
        """
        Sets the value of the parameter.

        Args:
            value (bool): The boolean value to set.

        Raises:
            ValueError: If the value is not a boolean.
        """
        if not isinstance(value, (bool, np.bool_)):
            raise ValueError(f"Value must be a boolean, but received type {type(value)}.")
        self.value = value

    def get_value(self) -> bool:
        """Returns the current value of the parameter."""
        return self.value

    def get_random_value(self, set_value: bool = False) -> bool:
        """
        Generates a random boolean value.

        Args:
            set_value (bool, optional): If True, sets the parameter's current value
                to the new random value. Defaults to False.

        Returns:
            bool: A random boolean (True or False).
        """
        r: bool = rd.choice([True, False])
        if set_value:
            self.set_value(r)
        return r

    def get_description(self) -> Dict[str, Any]:
        return {"type": "bool", "value": self.value}