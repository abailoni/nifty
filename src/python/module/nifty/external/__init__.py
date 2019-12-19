from __future__ import absolute_import
from . import _external as __external
from ._external import *

import numpy

__all__ = []
for key in __external.__dict__.keys():
    try:
        __external.__dict__[key].__module__='nifty.external'
    except:
        pass
    __all__.append(key)


def generate_opensimplex_noise(shape, seed=None, features_size=1., number_of_threads=-1):
    """
    :param features_size: Increase this value to get bigger spatially consistent features. Decrease it to get results
            similar to random Gaussian noise. Int or tuple
    """
    assert isinstance(shape, (list, tuple))
    if isinstance(shape, tuple):
        shape = list(shape)
    ndims = len(shape)

    if seed is None:
        seed = int(numpy.random.randint(-1000000000, 1000000000))

    if isinstance(features_size, float):
        features_size = numpy.ones(ndims, dtype="float32") * features_size
    else:
        assert isinstance(features_size, (numpy.ndarray, tuple, list))
        if not isinstance(features_size, numpy.ndarray):
            features_size = numpy.array(features_size)
        assert features_size.shape[0] == ndims

    features_size = features_size.astype('float64')
    assert all(features_size > 0.)


    output = evaluateSimplexNoiseOnArray_impl(
        shape=shape,
        seed=seed,
        featureSize=features_size,
        numberOfThreads=number_of_threads
    )

    return output