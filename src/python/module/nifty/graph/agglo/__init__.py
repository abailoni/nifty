from __future__ import absolute_import
from . import _agglo as __agglo
from ._agglo import *

import numpy

__all__ = []
for key in __agglo.__dict__.keys():
    __all__.append(key)
    try:
        __agglo.__dict__[key].__module__='nifty.graph.agglo'
    except:
        pass

from ...tools import makeDense as __makeDense


def updateRule(name, **kwargs):
    if name in ['max', 'single_linkage']:
        return MaxSettings()
    elif name in ['mutex_watershed', 'abs_max']:
        return MutexWatershedSettings()
    elif name in ['min', 'complete_linkage']:
        return MinSettings()
    elif name == 'sum':
        return SumSettings()
    elif name in ['mean', 'average', 'avg']:
        return ArithmeticMeanSettings()
    elif name in ['gmean', 'generalized_mean']:
        p = kwargs.get('p',1.0)
        return GeneralizedMeanSettings(p=float(p))
    elif name in ['smax', 'smooth_max']:
        p = kwargs.get('p',0.0)
        return SmoothMaxSettings(p=float(p))
    elif name in ['rank','quantile', 'rank_order']:
        q = kwargs.get('q',0.5)
        numberOfBins = kwargs.get('numberOfBins',40)
        return RankOrderSettings(q=float(q), numberOfBins=int(numberOfBins))
    else:
        return NotImplementedError("not yet implemented")

def get_GASP_policy(graph,
                    signed_edge_weights,
                    linkage_criteria = 'mean',
                    linkage_criteria_kwargs = None,
                    add_cannot_link_constraints= False,
                    edge_sizes = None,
                    is_mergeable_edge = None,
                    node_sizes = None,
                    size_regularizer = 0.0,
                    number_of_nodes_to_stop = 1
                    ):
    linkage_criteria_kwargs = {} if linkage_criteria_kwargs is None else linkage_criteria_kwargs
    parsed_rule = updateRule(linkage_criteria, **linkage_criteria_kwargs)

    edge_sizes = numpy.ones_like(signed_edge_weights) if edge_sizes is None else edge_sizes
    is_mergeable_edge = numpy.ones_like(signed_edge_weights) if is_mergeable_edge is None else is_mergeable_edge
    node_sizes = numpy.ones(graph.numberOfNodes ,dtype='float32') if node_sizes is None else node_sizes

    return gaspClusterPolicy(graph=graph,
                             signedWeights=signed_edge_weights,
                             isMergeEdge=is_mergeable_edge,
                             edgeSizes=edge_sizes,
                             nodeSizes=node_sizes,
                             updateRule0=parsed_rule,
                             numberOfNodesStop=number_of_nodes_to_stop,
                             sizeRegularizer=size_regularizer,
                             addNonLinkConstraints=add_cannot_link_constraints)


get_GASP_policy.__doc__ = """
GASP: generalized agglomeration for signed graph partition

Accepted update rules:
 - 'mean'
 - 'max' (single linkage)
 - 'min' (complete linkage)
 - 'MutexWatershed' (abs-max)
 - 'sum'
 - {name: 'rank', q=0.5, numberOfBins=40}
 - {name: 'generalized_mean', p=2.0}   # 1.0 is mean
 - {name: 'smooth_max', p=2.0}   # 0.0 is mean
 """











# def fixationClusterPolicy(graph, 
#     mergePrios=None,
#     notMergePrios=None,
#     edgeSizes=None,
#     isLocalEdge=None,
#     updateRule0="smooth_max",
#     updateRule1="smooth_max",
#     p0=float('inf'),
#     p1=float('inf'),
#     zeroInit=False):
    
#     if isLocalEdge is None:
#         raise RuntimeError("`isLocalEdge` must not be none")

#     if mergePrios is None and if notMergePrios is  None:
#         raise RuntimeError("`mergePrios` and `notMergePrios` cannot be both None")

#     if mergePrio is None:
#         nmp = notMergePrios.copy()
#         nmp -= nmp.min()
#         nmp /= nmp.max()
#         mp = 1.0 = nmp
#     elif notMergePrios is None:
#         mp = notMergePrios.copy()
#         mp -= mp.min()
#         mp /= mp.max()
#         nmp = 1.0 = mp
#     else:
#         mp = mergePrios
#         nmp = notMergePrios

#     if edgeSizes is None:
#         edgeSizes = numpy.ones(graph.edgeIdUpperBound+1)




#     if(updateRule0 == "histogram_rank" and updateRule1 == "histogram_rank"):
#         return nifty.graph.agglo.rankFixationClusterPolicy(graph=graph,
#             mergePrios=mp, notMergePrios=nmp,
#                         edgeSizes=edgeSizes, isMergeEdge=isLocalEdge,
#                         q0=p0, q1=p1, zeroInit=zeroInit)
#     elif(updateRule0 in ["smooth_max","generalized_mean"] and updateRule1 in ["smooth_max","generalized_mean"]):
        

#         return  nifty.graph.agglo.generalizedMeanFixationClusterPolicy(graph=g,
#                         mergePrios=mp, notMergePrios=nmp,
#                         edgeSizes=edgeSizes, isMergeEdge=isLocalEdge,
#                         p0=p0, p1=p1, zeroInit=zeroInit)

       













def sizeLimitClustering(graph, nodeSizes, minimumNodeSize, 
                        edgeIndicators=None,edgeSizes=None, 
                        sizeRegularizer=0.001, gamma=0.999,
                        makeDenseLabels=False):

    s = graph.edgeIdUpperBound + 1

    def rq(data):
        return numpy.require(data, 'float32')

    nodeSizes  = rq(nodeSizes)

    if edgeIndicators is None:
        edgeIndicators = numpy.ones(s,dtype='float32')
    else:
        edgeIndicators = rq(edgeIndicators)

    if edgeSizes is None:
        edgeSizes = numpy.ones(s,dtype='float32')
    else:
        edgeSizes = rq(edgeSizes)



    cp =  minimumNodeSizeClusterPolicy(graph, edgeIndicators=edgeIndicators, 
                                              edgeSizes=edgeSizes,
                                              nodeSizes=nodeSizes,
                                              minimumNodeSize=float(minimumNodeSize),
                                              sizeRegularizer=float(sizeRegularizer),
                                              gamma=float(gamma))

    agglo = agglomerativeClustering(cp)

    agglo.run()
    labels = agglo.result()

    if makeDenseLabels:
        labels = __makeDense(labels)

    return labels;




def ucmFeatures(graph, edgeIndicators, edgeSizes, nodeSizes, 
                sizeRegularizers = numpy.arange(0.1,1,0.1) ):
    
    def rq(data):
        return numpy.require(data, 'float32')
 
    edgeIndicators = rq(edgeIndicators)

    if edgeSizes is None:
        edgeSizes = numpy.ones(s,dtype='float32')
    else:
        edgeSizes = rq(edgeSizes)


    if nodeSizes is None:
        nodeSizes = numpy.ones(s,dtype='float32')
    else:
        nodeSizes = rq(nodeSizes)

    fOut = []
    # policy
    for sr in sizeRegularizers:

        sr = float(sr)
        cp = edgeWeightedClusterPolicyWithUcm(graph=graph, edgeIndicators=edgeIndicators,
                edgeSizes=edgeSizes, nodeSizes=nodeSizes, sizeRegularizer=sr)


        agglo = agglomerativeClustering(cp)



        hA = agglo.runAndGetDendrogramHeight()[:,None]
        hB = agglo.ucmTransform(cp.edgeIndicators)[:,None]

        fOut.extend([hA,hB])

    return numpy.concatenate(fOut, axis=1)

