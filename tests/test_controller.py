from ember_inspect.controller import read_controller_params, write_controller_params, reset_controller_params, _get_default_controller_params

# NOTE: The controller params below do not reflect the actual controller params
# used by the Ember model. They are just for testing.

def test_read_write_controller_params():
    write_controller_params({"test": "test"})
    assert read_controller_params() == {"test": "test"}

def test_reset_controller_params():
    write_controller_params({"test": "test"})
    reset_controller_params()
    assert read_controller_params() == _get_default_controller_params()