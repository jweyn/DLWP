#
# Copyright (c) 2017-18 Jonathan Weyn <jweyn@uw.edu>
#
# See the file LICENSE for your rights.
#

"""
DLWP utilities.
"""

import pickle
import tempfile
from importlib import import_module
from copy import copy
import keras.models


# ==================================================================================================================== #
# General utility functions
# ==================================================================================================================== #

def make_keras_picklable():
    """
    Thanks to http://zachmoshe.com/2017/04/03/pickling-keras-models.html
    """

    def __getstate__(self):
        model_str = ""
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            keras.models.save_model(self, fd.name, overwrite=True)
            model_str = fd.read()
        d = {'model_str': model_str}
        return d

    def __setstate__(self, state):
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            fd.write(state['model_str'])
            fd.flush()
            model = keras.models.load_model(fd.name)
        self.__dict__ = model.__dict__

    cls = keras.models.Model
    cls.__getstate__ = __getstate__
    cls.__setstate__ = __setstate__


def get_object(module_class):
    """
    Given a string with a module class name, it imports and returns the class.
    This function (c) Tom Keffer, weeWX; modified by Jonathan Weyn.
    """
    # Split the path into its parts
    parts = module_class.split('.')
    # Get the top level module
    module = parts[0]  # '.'.join(parts[:-1])
    # Import the top level module
    mod = __import__(module)
    # Recursively work down from the top level module to the class name.
    # Be prepared to catch an exception if something cannot be found.
    try:
        for part in parts[1:]:
            module = '.'.join([module, part])
            # Import each successive module
            __import__(module)
            mod = getattr(mod, part)
    except ImportError as e:
        # Can't find a recursive module. Give a more informative error message:
        raise ImportError("'%s' raised when searching for %s" % (str(e), module))
    except AttributeError:
        # Can't find the last attribute. Give a more informative error message:
        raise AttributeError("Module '%s' has no attribute '%s' when searching for '%s'" %
                             (mod.__name__, part, module_class))

    return mod


def get_from_class(module_name, class_name):
    """
    Given a module name and a class name, return an object corresponding to the class retrieved as in
    `from module_class import class_name`.

    :param module_name: str: name of module (may have . attributes)
    :param class_name: str: name of class
    :return: object pointer to class
    """
    mod = __import__(module_name, fromlist=[class_name])
    class_obj = getattr(mod, class_name)
    return class_obj


def get_classes(module_name):
    """
    From a given module name, return a dictionary {class_name: class_object} of its classes.

    :param module_name: str: name of module to import
    :return: dict: {class_name: class_object} pairs in the module
    """
    module = import_module(module_name)
    classes = {}
    for key in dir(module):
        if isinstance(getattr(module, key), type):
            classes[key] = get_from_class(module_name, key)
    return classes


def save_model(model, file_name, history=None):
    """
    Saves a class instance with a 'model' attribute to disk. Creates two files: one pickle file containing no model
    saved as ${file_name}.pkl and one for the model saved as ${file_name}.keras. Use the `load_model()` method to load
    a model saved with this method.

    :param model: model instance (with a 'model' attribute) to save
    :param file_name: str: base name of save files
    :param history: history from Keras fitting, or None
    :return:
    """
    model.model.save('%s.keras' % file_name)
    model_copy = copy(model)
    model_copy.model = None
    with open('%s.pkl' % file_name, 'wb') as f:
        pickle.dump(model_copy, f, protocol=pickle.HIGHEST_PROTOCOL)
    if history is not None:
        with open('%s.history' % file_name, 'wb') as f:
            pickle.dump(history.history, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_model(file_name, history=False, custom_objects=None):
    """
    Loads a model saved to disk with the `save_model()` method.

    :param file_name: str: base name of save files
    :param history: bool: if True, loads the history file along with the model
    :param custom_objects: dict: any custom functions or classes to be included when Keras loads the model. There is
        no need to add objects in DLWP.custom as those are added automatically.
    :return: model [, dict]: loaded object [, dictionary of training history]
    """
    with open('%s.pkl' % file_name, 'rb') as f:
        model = pickle.load(f)
    custom_objects = custom_objects or {}
    custom_objects.update(get_classes('DLWP.custom'))
    model.model = keras.models.load_model('%s.keras' % file_name, custom_objects=custom_objects, compile=True)
    if history:
        with open('%s.history' % file_name, 'rb') as f:
            h = pickle.load(f)
        return model, h
    else:
        return model


def save_torch_model(model, file_name, history=None):
    """
    Saves a DLWPTorchNN model to disk. Creates two files: one pickle file containing the DLWPTorchNN wrapper, saved as
    ${file_name}.pkl, and one for the model saved as ${file_name}.torch. Use the `load_torch_model()` method to load
    a model saved with this method.

    :param model: DLWPTorchNN or other torch model to save
    :param file_name: str: base name of save files
    :param history: history of model to save; optional
    :return:
    """
    import torch
    torch.save(model.model, '%s.torch' % file_name)
    model_copy = copy(model)
    model_copy.model = None
    with open('%s.pkl' % file_name, 'wb') as f:
        pickle.dump(model_copy, f, protocol=pickle.HIGHEST_PROTOCOL)
    if history is not None:
        with open('%s.history' % file_name, 'wb') as f:
            pickle.dump(history, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_torch_model(file_name, history=False):
    """
    Loads a DLWPTorchNN or other model using Torch saved to disk with the `save_torch_model()` method.

    :param file_name: str: base name of save files
    :param history: bool: if True, loads the history file along with the model
    :return: model [, dict]: loaded object [, dictionary of training history]\
    """
    import torch
    with open('%s.pkl' % file_name, 'rb') as f:
        model = pickle.load(f)
    model.model = torch.load('%s.torch' % file_name)
    model.model.eval()
    if history:
        with open('%s.history' % file_name, 'rb') as f:
            h = pickle.load(f)
        return model, h
    else:
        return model
