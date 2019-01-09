from __future__ import print_function, absolute_import

from .SoftmaxNeigLoss import SoftmaxNeigLoss
from .KNNSoftmax import KNNSoftmax
from .myKNNSoftmax import myKNNSoftmax
from .NeighbourLoss import NeighbourLoss
from .triplet import TripletLoss
from .CenterTriplet import CenterTripletLoss
from .GaussianMetric import GaussianMetricLoss
from .HistogramLoss import HistogramLoss
from .BatchAll import BatchAllLoss
from .NeighbourLoss import NeighbourLoss
from .DistanceMatchLoss import DistanceMatchLoss
from .NeighbourHardLoss import NeighbourHardLoss
from .DistWeightLoss import DistWeightLoss
from .BinDevianceLoss import BinDevianceLoss
from .BinBranchLoss import BinBranchLoss
from .MarginDevianceLoss import MarginDevianceLoss
from .MarginPositiveLoss import MarginPositiveLoss
from .ContrastiveLoss import ContrastiveLoss
from .DistWeightContrastiveLoss import DistWeightContrastiveLoss
from .DistWeightDevianceLoss import DistWeightBinDevianceLoss
from .DistWeightDevBranchLoss import DistWeightDevBranchLoss
from .DistWeightNeighbourLoss import DistWeightNeighbourLoss
from .BDWNeighbourLoss import BDWNeighbourLoss
from .EnsembleDWNeighbourLoss import EnsembleDWNeighbourLoss



__factory = {
    'softneig': SoftmaxNeigLoss,
    'knnsoftmax': KNNSoftmax,
    'myknnsoftmax': myKNNSoftmax,
    'neighbour': NeighbourLoss,
    'triplet': TripletLoss,
    'histogram': HistogramLoss,
    'gaussian': GaussianMetricLoss,
    'batchall': BatchAllLoss,
    'neighard': NeighbourHardLoss,
    'bin': BinDevianceLoss,
    'binbranch': BinBranchLoss,
    'margin': MarginDevianceLoss,
    'positive': MarginPositiveLoss,
    'con': ContrastiveLoss,
    'distweight': DistWeightLoss,
    'distance_match': DistanceMatchLoss,
    'dwcon': DistWeightContrastiveLoss,
    'dwdev': DistWeightBinDevianceLoss,
    'dwneig': DistWeightNeighbourLoss,
    'dwdevbranch': DistWeightDevBranchLoss,
    'bdwneig': BDWNeighbourLoss,
    'edwneig': EnsembleDWNeighbourLoss,
}


def names():
    return sorted(__factory.keys())


def create(name, *args, **kwargs):
    """
    Create a loss instance.

    Parameters
    ----------
    name : str
        the name of loss function
    """
    if name not in __factory:
        raise KeyError("Unknown loss:", name)
    return __factory[name](*args, **kwargs)



def names():
    return sorted(__factory.keys())


def create(name, *args, **kwargs):
    """
    Create a loss instance.

    Parameters
    ----------
    name : str
        the name of loss function
    """
    if name not in __factory:
        raise KeyError("Unknown loss:", name)
    return __factory[name]( *args, **kwargs)