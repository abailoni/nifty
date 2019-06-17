from __future__ import absolute_import,print_function

import numpy as np
from ._segmentation import *
import time


def get_valid_edges(shape, offsets, number_of_attractive_channels,
                    strides, randomize_strides, mask=None):
    # compute valid edges, i.e. the ones not going out of the image boundaries
    ndim = len(offsets[0])
    image_shape = shape[1:]
    valid_edges = np.ones(shape, dtype=bool)
    for i, offset in enumerate(offsets):
        for j, o in enumerate(offset):
            inv_slice = slice(0, -o) if o < 0 else slice(max(image_shape[j] - o,0), image_shape[j])
            invalid_slice = (i, ) + tuple(slice(None) if j != d else inv_slice
                                          for d in range(ndim))
            valid_edges[invalid_slice] = 0

    # mask additional edges if we have strides
    if strides is not None:
        assert len(strides) == ndim
        if randomize_strides:
            stride_factor = 1 / np.prod(strides)
            stride_edges = np.random.rand(*valid_edges.shape) < stride_factor
            stride_edges[:number_of_attractive_channels] = 1
            valid_edges = np.logical_and(valid_edges, stride_edges)
        else:
            stride_edges = np.zeros_like(valid_edges, dtype='bool')
            stride_edges[:number_of_attractive_channels] = 1
            valid_slice = (slice(number_of_attractive_channels, None),) +\
                tuple(slice(None, None, stride) for stride in strides)
            stride_edges[valid_slice] = 1
            valid_edges = np.logical_and(valid_edges, stride_edges)

    # if we have an external mask, mask all transitions to and within that mask
    if mask is not None:
        assert mask.shape == image_shape, "%s, %s" % (str(mask.shape), str(image_shape))
        assert mask.dtype == np.dtype('bool'), str(mask.dtype)
        # mask transitions to mask
        transition_to_mask, _ = compute_affinities(mask, offsets)
        transition_to_mask = transition_to_mask == 0
        valid_edges[transition_to_mask] = False
        # mask within mask
        valid_edges[:, mask] = False

    return valid_edges

def get_sorted_flat_indices_and_valid_edges(weights, offsets, number_of_attractive_channels,
                                            strides=None, randomize_strides=False, invert_repulsive_weights=True,
                                            bias_cut=0.):
    ndim = len(offsets[0])
    assert all(len(off) == ndim for off in offsets)
    image_shape = weights.shape[1:]

    valid_edges = get_valid_edges(weights.shape, offsets, number_of_attractive_channels,
                                  strides, randomize_strides)
    if invert_repulsive_weights:
        weights[number_of_attractive_channels:] *= -1
        weights[number_of_attractive_channels:] += 1
    weights[:number_of_attractive_channels] += bias_cut

    masked_weights = np.ma.masked_array(weights, mask=np.logical_not(valid_edges))

    tick = time.time()
    sorted_flat_indices = np.argsort(masked_weights, axis=None)[::-1]
    tock = time.time()
    print("Sorted edges in {}s".format(tock-tick))

    return valid_edges.ravel().astype('bool'), sorted_flat_indices.astype('uint64')

def run_mws(sorted_flat_indices,
                valid_edges,
                        offsets,
                        number_of_attractive_channels,
                        image_shape,
                        algorithm='kruskal'):
    assert algorithm in ('kruskal', 'divisive'), "Unsupported algorithm, %s" % algorithm
    if algorithm == 'kruskal':
        labels = compute_mws_segmentation_impl(sorted_flat_indices,
                                               valid_edges.ravel(),
                                               offsets,
                                               number_of_attractive_channels,
                                               image_shape)
    else:
        labels = compute_divisive_mws_segmentation_impl(sorted_flat_indices,
                                                    valid_edges.ravel(),
                                                    offsets,
                                                    number_of_attractive_channels,
                                                    image_shape)




def compute_mws_segmentation(weights, offsets, number_of_attractive_channels,
                             strides=None, randomize_strides=False, invert_repulsive_weights=True,
                             bias_cut=0., mask=None,
                             algorithm='kruskal'):
    assert algorithm in ('kruskal', 'prim'), "Unsupported algorithm, %s" % algorithm
    ndim = len(offsets[0])
    assert all(len(off) == ndim for off in offsets)
    image_shape = weights.shape[1:]

    # we assume that we get a 'valid mask', i.e. a mask where valid regions are set true
    # and invalid regions are set to false.
    # for computation, we need the opposite though
    inv_mask = None if mask is None else np.logical_not(mask)
    valid_edges = get_valid_edges(weights.shape, offsets, number_of_attractive_channels,
                                  strides, randomize_strides, inv_mask)

    # FIXME: double check if it is still necessary to invert weights (deleted from master)
    weights = np.copy(weights)
    if invert_repulsive_weights:
        weights[number_of_attractive_channels:] *= -1
        weights[number_of_attractive_channels:] += 1
    weights[:number_of_attractive_channels] += bias_cut

    if algorithm == 'kruskal' or algorithm == 'divisive':
        # sort and flatten weights
        # ignore masked weights during sorting
        masked_weights = np.ma.masked_array(weights, mask=np.logical_not(valid_edges))

        tick = time.time()
        sorted_flat_indices = np.argsort(masked_weights, axis=None)[::-1]
        tock = time.time()
        print("Sorted edges in {}s".format(tock-tick))

        # sorted_flat_indices = np.argsort(weights, axis=None)[::-1]
        if algorithm == 'kruskal':
            labels = compute_mws_segmentation_impl(sorted_flat_indices,
                                               valid_edges.ravel(),
                                               offsets,
                                               number_of_attractive_channels,
                                               image_shape)
        elif algorithm == 'divisive':
            labels = compute_divisive_mws_segmentation_impl(sorted_flat_indices,
                                                   valid_edges.ravel(),
                                                   offsets,
                                                   number_of_attractive_channels,
                                                   image_shape)
    else:
        labels = compute_mws_prim_segmentation_impl(weights.ravel(),
                                                    valid_edges.ravel(),
                                                    offsets,
                                                    number_of_attractive_channels,
                                                    image_shape)

    labels = labels.reshape(image_shape)
    # if we had an external mask, make sure it is mapped to zero
    if mask is not None:
        # increase labels by 1, so we don't merge anything with the mask
        labels += 1
        labels[inv_mask] = 0
    return labels
#
# # \\\\\\\\\\\\\\\\\\\\\
# # Previous functions:
#
#
# # TODO: add these dependencies in the conda-recipe
# # FIXME: ImportError: cannot import name 'imresize' from 'scipy.misc' (/home/abailoni_local/miniconda3/envs/GASP/lib/python3.7/site-packages/scipy/misc/__init__.py)
#
#
# from .. import graph
# from .. import filters
#
# from skimage.feature import peak_local_max as __peak_local_max
# import skimage.segmentation
# from scipy.misc import imresize as __imresize
# from scipy.ndimage import zoom as __zoom
# import scipy.ndimage
# import numpy
#
#
#
# def slic(image, nSegments, components):
#     """ same as skimage.segmentation.slic """
#     return skimage.segmentation.slic(image, n_segments=nSegments,
#                                      compactness=compactness)
#
#
# def seededWatersheds(heightMap, seeds=None, method="node_weighted", acc="max"):
#     """Seeded watersheds segmentation
#
#     Get a segmentation via seeded watersheds.
#     This is a high level wrapper around
#     :func:`nifty.graph.nodeWeightedWatershedsSegmentation`
#     and :func:`nifty.graph.nodeWeightedWatershedsSegmentation`.
#
#
#     Args:
#         heightMap (numpy.ndarray) : height / evaluation map
#         seeds (numpy.ndarray) : Seeds as non zero elements in the array.
#             (default: {nifty.segmentation.localMinimaSeeds(heightMap)})
#         method (str): Algorithm type can be:
#
#             *   "node_weighted": ordinary node weighted watershed
#             *   "edge_weighted": edge weighted watershed (minimum spanning tree)
#
#             (default: {"max"})
#
#         acc (str): If method is "edge_weighted", one needs to specify how
#             to convert the heightMap into an edgeMap.
#             This parameter specificities this method.
#             Allow values are:
#
#             *   'min' : Take the minimum value of the endpoints of the edge
#             *   'max' : Take the minimum value of the endpoints of the edge
#             *   'sum' : Take the sum  of the values of the endpoints of the edge
#             *   'prod' : Take the product of the values of the endpoints of the edge
#             *   'interpixel' : Take the value of the image at the interpixel
#                 coordinate in between the two endpoints.
#                 To do this the image is resampled to have shape :math: `2 \cdot shape -1 `
#
#             (default: {"max"})
#
#     Returns:
#         numpy.ndarray : the segmentation
#
#     Raises:
#         RuntimeError: [description]
#     """
#     if seeds is None:
#         seeds = localMinimaSeeds(heightMap)
#
#     hshape = heightMap.shape
#     sshape = seeds.shape
#     shape = sshape
#
#
#     ishape = [2*s -1 for s in shape]
#     gridGraph = graph.gridGraph(shape)
#
#     # node watershed
#     if method == "node_weighted":
#         assert hshape == sshape
#         seg = graph.nodeWeightedWatershedsSegmentation(graph=gridGraph, seeds=seeds.ravel(),
#                                                        nodeWeights=heightMap.ravel())
#         seg = seg.reshape(shape)
#
#     elif method == "edge_weighted":
#         if acc != 'interpixel':
#             assert hshape == sshape
#             gridGraphEdgeStrength = gridGraph.imageToEdgeMap(heightMap, mode=acc)
#
#         else:
#             if(hshape == shape):
#                 iHeightMap = __imresize(heightMap, ishape, interp='bicubic')
#                 gridGraphEdgeStrength = gridGraph.imageToEdgeMap(iHeightMap, mode=acc)
#             elif(hshape == ishape):
#                 gridGraphEdgeStrength = gridGraph.imageToEdgeMap(heightMap, mode=acc)
#             else:
#                 raise RuntimeError("height map has wrong shape")
#
#         seg = graph.edgeWeightedWatershedsSegmentation(graph=gridGraph, seeds=seeds.ravel(),
#                                                        edgeWeights=gridGraphEdgeStrength)
#         seg = seg.reshape(shape)
#
#
#     return seg
#
# def distanceTransformWatersheds(pmap, preBinarizationMedianRadius=1, threshold = 0.5, preSeedSigma=0.75):
#     """Superpixels for neuro data as in http://brainiac2.mit.edu/isbi_challenge/
#
#     Use raw data and membrane probability maps to
#     generate a over-segmentation suitable for neuro data
#
#     Args:
#         pmap (numpy.ndarray): Membrane probability in [0,1].
#         preBinarizationMedianRadius (int) : Radius of
#             diskMedian filter applied to the probability map
#             before binarization. (default:{1})
#         threshold (float) : threshold to binarize
#             probability map  before applying
#             the distance transform (default: {0.5})
#         preSeedSigma (float) : smooth the distance
#             transform image before getting the seeds.
#
#     Raises:
#         RuntimeError: if applied to data with wrong dimensionality
#     """
#     if pmap.ndim != 2:
#         raise RuntimeError("Currently only implemented for 2D data")
#
#
#
#
#     # pre-process pmap  / smooth pmap
#     if preBinarizationMedianRadius >= 1 :
#         toBinarize = filters.diskMedian(pmap, radius=preBinarizationMedianRadius)
#         toBinarize -= toBinarize.min()
#         toBinarize /= toBinarize.max()
#     else:
#         toBinarize = pmap
#
#
#     # computing the distance transform inside and outside
#     b1 = toBinarize < threshold
#
#     #b0 = b1 == False
#
#     dt1 = scipy.ndimage.morphology.distance_transform_edt(b1)
#     #dt0 = scipy.ndimage.morphology.distance_transform_edt(b0)
#
#     if preSeedSigma > 0.01:
#         toSeedOn = filters.gaussianSmoothing(dt1, preSeedSigma)
#     else:
#         toSeedOn = dt1
#     # find the seeds
#     seeds  = localMaximaSeeds(toSeedOn)
#
#     # compute  growing map
#     a = filters.gaussianSmoothing(pmap, 0.75)
#     b = filters.gaussianSmoothing(pmap, 3.00)
#     c = filters.gaussianSmoothing(pmap, 9.00)
#     d =  growMapB = numpy.exp(-0.2*dt1)
#     growMap = (7.0*a + 4.0*b + 2.0*c + 1.0*d)/14.0
#
#
#     # grow regions
#     seg = seededWatersheds(growMap, seeds=seeds,
#                            method='edge_weighted',  acc='interpixel')
#     return seg
#
# def localMinima(image):
#     """get the local minima of an image
#
#     Get the local minima wrt a 4-neighborhood on an image.
#     For a plateau, all pixels of this plateau are marked
#     as minimum pixel.
#
#     Args:
#         image (numpy.ndarray): the input image
#
#     Returns:
#         (numpy.ndarray) : array which is 1 the minimum 0 elsewhere.
#
#     """
#     if image.ndim != 2:
#         raise RuntimeError("localMinima is currently only implemented for 2D images")
#     return localMaxima(-1.0*image)
#
# def localMaxima(image):
#     """get the local maxima of an image
#
#     Get the local maxima wrt a 4-neighborhood on an image.
#     For a plateau, all pixels of this plateau are marked
#     as maximum pixel.
#
#     Args:
#         image (numpy.ndarray): the input image
#
#     Returns:
#         (numpy.ndarray) : array which is 1 the maximum 0 elsewhere.
#
#     """
#     if image.ndim != 2:
#         raise RuntimeError("localMaxima is currently only implemented for 2D images")
#
#     return  __peak_local_max(image, exclude_border=False, indices=False)
#
# def connectedComponents(labels, dense=True, ignoreBackground=False):
#     """get connected components of a label image
#
#     Get connected components of an image w.r.t.
#     a 4-neighborhood .
#     This is a high level wrapper for
#         :func:`nifty.graph.connectedComponentsFromNodeLabels`
#
#     Args:
#         labels (numpy.ndarray):
#         dense (bool): should the return labeling be dense (default: {True})
#         ignoreBackground (bool): should values of zero be excluded (default: {False})
#
#     Returns:
#         [description]
#         [type]
#     """
#     shape = labels.shape
#     gridGraph = graph.gridGraph(shape)
#
#     ccLabels = graph.connectedComponentsFromNodeLabels(gridGraph,
#                                                        labels.ravel(), dense=dense,
#                                                        ignoreBackground=bool(ignoreBackground))
#
#     return ccLabels.reshape(shape)
#
# def localMinimaSeeds(image):
#     """Get seed from local minima
#
#     Get seeds by running connected components
#     on the local minima.
#     This is a high level wrapper around
#     :func:`nifty.segmentation.localMinima`
#     and :func:`nifty.segmentation.connectedComponents`
#
#     Args:
#         image: [description]
#
#     Returns:
#         [description]
#         [type]
#
#     Raises:
#         RuntimeError: [description]
#     """
#     return localMaximaSeeds(-1.0 * image)
#
# def localMaximaSeeds(image):
#     """Get seed from local maxima
#
#     Get seeds by running connected components
#     on the local maxima.
#     This is a high level wrapper around
#     :func:`nifty.segmentation.localMinima`
#     and :func:`nifty.segmentation.connectedComponents`
#
#     Args:
#         image: [description]
#
#     Returns:
#         [description]
#         [type]
#
#     Raises:
#         RuntimeError: [description]
#     """
#     if image.ndim != 2:
#         raise RuntimeError("localMaximaSeeds is currently only implemented for 2D images")
#
#     lm = localMaxima(image)
#     cc = connectedComponents(lm, dense=True, ignoreBackground=True)
#     return cc
#
# def markBoundaries(image, segmentation, color=None, thin=True):
#     """Mark the boundaries in an image
#
#     Mark boundaries in an image.
#
#     Warning:
#
#         The returned image shape is twice as large
#         as the input if this is True.
#
#     Args:
#         image:  the input image
#         segmentation:  the segmentation
#         color (tuple) : the edge color(default: {(0,0,0)})
#         thin (bool) : IF true, the image is interpolated and
#             the boundaries are marked in the interpolated
#             image. This will make the output twice as large.
#     Returns:
#         (numpy.ndarray) : image with marked boundaries. Note that
#             result image has twice as large shape as the input if thin is True.
#     """
#     if color is None:
#         color = (0,0,0)
#     if thin:
#         shape = segmentation.shape
#         img2 =   __imresize(image, [2*s for s in shape])#, interp='nearest')
#         #img2 = __zoom(segmentation.astype('float32'), 2, order=1)
#         seg2 = __zoom(segmentation.astype('float32'), 2, order=0)
#         seg2 = seg2.astype('uint32')
#
#         return skimage.segmentation.mark_boundaries(img2, seg2.astype('uint32'), color=color)
#     else:
#         return skimage.segmentation.mark_boundaries(image, segmentation.astype('uint32'), color=color)
#
# def segmentOverlay(image, segmentation, beta=0.5, zeroToZero=False, showBoundaries=True, color=None, thin=True):
#
#     cmap = numpy.random.rand (int(segmentation.max()+1), 3)
#
#     if zeroToZero:
#         cmap[0,:] = 0
#     cSeg = numpy.take(cmap, numpy.require(segmentation,dtype='int64'),axis=0)
#
#     imgCp = image.astype('float32')
#     if imgCp.ndim != 3:
#         imgCp  = numpy.concatenate([imgCp[:,:,None]]*3,axis=2)
#
#
#     mi = imgCp.min()
#     ma = imgCp.max()
#
#     if(ma-mi > 0.000001):
#         imgCp -= mi
#         imgCp /= (ma - mi)
#
#     overlayImg =  (1.0-beta)*imgCp + (beta)*cSeg
#
#     if showBoundaries:
#         return markBoundaries(overlayImg, segmentation, color=color, thin=thin)
#     else:
#         return overlayImg
#
#
#
#
#
# def randomColormap(size=10000, zeroToZero=False):
#     try:
#         from matplotlib.colors import ListedColormap as __ListedColormap
#     except ImportError:
#         print("matplotlib is needed to use a random-colormap")
#         raise ImportError
#     cmap = numpy.random.rand (int(size),3)
#     if zeroToZero:
#         cmap[0,:] = 0
#     cmap = __ListedColormap(cmap)
#     return cmap
#
#
#
#
#
#
