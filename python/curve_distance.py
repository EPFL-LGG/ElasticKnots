import numpy as np
from scipy import fftpack
from scipy import interpolate
from copy import copy

import sys; sys.path.append("../..")
import ElasticRods
from elastic_rods import *

from elastic_knots import *

def compute_curvature(pts_crv):
    """ Compute curvature along a closed curve. Note : currently unused
    Args:
        pts_crv (np.array) : positions of the curve points,first and last point must overlap
    """
    # assert np.allclose(pts_crv[0], pts_crv[-1], rtol=1.0e-5), "First and last points must overlap for closed curves"
    diff = pts_crv[1:] - pts_crv[:-1]
    tangents = diff / np.linalg.norm(diff, axis=1).reshape(-1,1)
    normals = np.cross(tangents, np.roll(tangents, 1, axis=0), axis=1)
    cos_angle = np.clip(np.sum(tangents * np.roll(tangents, 1, axis=0), axis=1), -1.0, 1.0)
    sin_angle = np.clip(np.sum(normals * normals, axis=1), -1.0, 1.0)
    curvature_angles = np.arctan2(sin_angle, cos_angle)

    return curvature_angles

def curve_dist(c1,c2):
    """ Simple curve distance based on curvature profile, unused
    """
    c1 = c1.reshape(-1,3)
    c2 = c2.reshape(-1,3)
    c1 = np.concatenate([c1,[c1[0]]])
    c2 = np.concatenate([c2,[c2[0]]])
    curv1 = compute_curvature(c1)
    curv2 = compute_curvature(c2)
    d = (curv1 - curv2)**2
    return np.sum(d)

def curvature_distance(rod1,rod2):
    pts1 = np.array(rod1.deformedPoints())
    pts2 = np.array(rod2.deformedPoints())
    curvature1 = compute_curvature(pts1)
    print(curvature1)
    curvature2 = compute_curvature(pts2)
    print(curvature2)
    similarity = curvature1.dot(curvature2) / np.linalg.norm(curvature1) / np.linalg.norm(curvature2)
    print(similarity)
    similarity = np.clip(similarity, a_min=0.0, a_max=1.0)
    return 1 - similarity

def energy_based_distance(rod1,rod2,metrics=None,reverse=False,mirror=False):
    e1 = rod1.energy()
    e2 = rod2.energy()
    return np.abs(e1-e2)

def rod_distance(rod1,rod2,metrics=['curvature', 'curv-times-tors'],reverse=False):
    similarity_measure = 1
    for metric in metrics:
        if metric == 'curvature':
            curvature1 = rod1.curvature(CurvatureDiscretizationType.Angle, pointwise=True)
            curvature2 = rod2.curvature(CurvatureDiscretizationType.Angle, pointwise=True)
            similarity = curvature1.dot(curvature2) / np.linalg.norm(curvature1) / np.linalg.norm(curvature2)
            similarity = np.clip(similarity, a_min=0.0, a_max=1.0)
            if reverse:
                sim = curvature1.dot(np.flip(curvature2)) / np.linalg.norm(curvature1) / np.linalg.norm(curvature2)
                similarity = max(similarity,sim)
            similarity_measure *= similarity 
        elif metric == 'torsion':
            torsion1 = compute_torsion(rod1)
            torsion2 = compute_torsion(rod2)
            similarity = torsion1.dot(torsion2) / np.linalg.norm(torsion1) / np.linalg.norm(torsion2)
            similarity = np.clip(similarity, a_min=0.0, a_max=1.0)
            if reverse:
                sim = torsion1.dot(np.flip(torsion2)) / np.linalg.norm(torsion1) / np.linalg.norm(torsion2)
                similarity = max(similarity,sim)
            similarity_measure *= similarity 
        elif metric == 'curv-times-tors':
            torsion1 = compute_torsion(rod1)   # per vertex
            torsion2 = compute_torsion(rod2)   # per vertex
            curvature1 = rod1.curvature(CurvatureDiscretizationType.Angle, pointwise=True)
            curvature2 = rod2.curvature(CurvatureDiscretizationType.Angle, pointwise=True)
            torsion_on_verts1 = (torsion1 + np.roll(torsion1, 1)) / 2
            torsion_on_verts2 = (torsion2 + np.roll(torsion2, 1)) / 2
            curv_times_tors1 = curvature1 * torsion_on_verts1
            curv_times_tors2 = curvature2 * torsion_on_verts2
            similarity = curv_times_tors1.dot(curv_times_tors2) / np.linalg.norm(curv_times_tors1) / np.linalg.norm(curv_times_tors2)
            similarity = np.clip(similarity, a_min=0.0, a_max=1.0)
            if reverse:
                sim = curv_times_tors1.dot(np.flip(curv_times_tors2)) / np.linalg.norm(curv_times_tors1) / np.linalg.norm(curv_times_tors2)
                similarity = max(similarity,sim)
            similarity_measure *= similarity 
        else:
            raise ValueError('Unknown metric')
    return 1 - similarity_measure


def compute_torsion(rod:PeriodicRod):
    """ Python version of the torsion function, added a few fixes
    Args:
        rod (PeriodicRod) : rod to compute the torsion
    """
    
    # Rod points
    points = rod.getDoFs()
    n_vertices = int((len(points) - 1) / 4)
    points = points[:3*n_vertices].reshape(-1,3)
    
    # Edges
    e = np.zeros((n_vertices,3))
    e[:-1] = points[1:] - points[:-1]
    e[-1] = points[0] - points[-1]
    
    # Binormals
    b = np.zeros((n_vertices,3))
    b[1:] = np.cross(e[1:], e[:-1])
    b[0] = np.cross(e[0], e[-1])
    n = np.linalg.norm(b,axis = 1).reshape(-1,1)
    b = b/n

    # Edge lengths
    l = np.linalg.norm(e,axis = 1)
    
    # Angles
    a = np.zeros(n_vertices)
    axis = e / np.linalg.norm(e,axis=1).reshape(-1,1)
    c = np.einsum("ij,ij->i",np.cross(b[1:],b[:-1]),axis[:-1])
    d = np.einsum("ij,ij->i",b[1:],b[:-1])
    a[:-1] = np.arctan2(c,d)
    a[-1] = np.arctan2(np.cross(b[0],b[-1]).dot(axis[-1]), b[0].dot(b[-1]))
    
    # There can be a sign issue for angles equal to pi (numerical fluctuations around 0 for the cos)
    idx = np.abs((np.abs(a) - np.pi)) < 1e-6
    a[idx] = np.abs(a[idx])
    
    torsion = a / l
    
    if np.all(np.abs(torsion) < 1e-6):
        torsion[:] = 1e-10  # torsion can be close to zero everywhere (e.g. circle) => numerical fluctuations can result in low similarity score
    
    return torsion

def compute_correlation_and_convolution(rod_list, metrics=['curvature', 'curv-times-tors'], discretization_type=CurvatureDiscretizationType.Angle, pointwise=True):
    
    def uniform_resample(x, y, n_samples, kind='linear'):
        assert(x.size == y.size)
        f = interpolate.interp1d(x, y, kind=kind)
        x_new = np.linspace(x[0], x[-1], n_samples)
        y_new = f(x_new)
        return x_new, y_new
    
    n_rods = len(rod_list)
    n_metrics = len(metrics)
    
    n_vertices_per_rod = [rod.numVertices() for rod in rod_list]
    nv_max = int(np.max(n_vertices_per_rod))
    n_samples = nv_max

    # Build array of metrics
    signals = np.zeros((n_rods, n_metrics, n_samples))
    resampling_kind = 'linear'
    for ri, rod in enumerate(rod_list):
        cum_edge_lengths = np.append(0, np.cumsum(np.array(rod.restLengths()[:-1])))
        signals_ri = []
        for metric in metrics:
            if metric == 'curvature':
                curvature = rod.curvature(discretization_type, pointwise=pointwise)
                _, y = uniform_resample(cum_edge_lengths, curvature, n_samples, kind=resampling_kind)
                signals_ri.append(y)
                
            elif metric == 'torsion':
                _, y = uniform_resample(cum_edge_lengths, rod.torsion(discretization_type, pointwise=pointwise), n_samples, kind=resampling_kind)
                signals_ri.append(y)
            elif metric == 'curv-times-tors':
                torsion = compute_torsion(rod)   # per vertex
                curvature = rod.curvature(discretization_type, pointwise=pointwise)
                torsion_on_verts = (torsion + np.roll(torsion, 1)) / 2
                curv_times_tors = curvature * torsion_on_verts
                _, y = uniform_resample(cum_edge_lengths, curv_times_tors, n_samples, kind=resampling_kind)
                signals_ri.append(y)
            else:
                raise ValueError('Unknown metric')
                
        signals_ri = np.array(signals_ri)
        signals[ri, :, :] = signals_ri
        
    # FFT
    signals_fft = fftpack.fft(signals)  # shape: len(metrics) x n_samples. Each row is ffted separately (not a multi-dim fft!)
    signals_fft_conj = signals_fft.conjugate()
    
    # Correlation with cyclic boundary conditions
    autocorr = np.abs(fftpack.ifft(signals_fft_conj*signals_fft))
    crosscorr = np.zeros((n_rods, n_rods, n_metrics, n_samples))
    convol = np.zeros((n_rods, n_rods, n_metrics, n_samples))
    for a in range(n_rods):
        for b in range(n_rods):
            if b < a:
                continue
            crosscorr_a_b = fftpack.ifft(signals_fft_conj[a, :, :]*signals_fft[b, :, :])
            crosscorr_sign = np.sign(np.real(crosscorr_a_b))
            crosscorr[a, b, :, :] = crosscorr_sign * np.abs(crosscorr_a_b)
            convol_a_b = fftpack.ifft(signals_fft[a, :, :]*signals_fft[b, :, :])
            convol_sign = np.sign(np.real(convol_a_b))
            convol[a, b, :, :] = convol_sign * np.abs(convol_a_b)
    
    for a in range(n_rods):  # copy upper trianglar part to lower triangular
        for b in range(n_rods):
            if b < a:
                crosscorr[a, b, :, :] = crosscorr[b, a, :, :]
                convol[a, b, :, :] = convol[b, a, :, :]
    return autocorr, crosscorr, convol


def compute_similarity_from_correlation(autocorrA, autocorrB, crosscorr, convol=None):
    """
    Compute the similarity of a rod pair given the (cross-)correlation of their chosen metrics.
    Input shape: [n_metrics, n_samples]
    """
    
    assert(autocorrA.shape[0] == autocorrB.shape[0] == crosscorr.shape[0])
    if convol is not None:
        assert(autocorrA.shape[0] == convol.shape[0])
    n_metrics = autocorrA.shape[0]
    
    def combine_metrics(computed_metrics, weights='ones'):
        "Compute the weighted product of different scores in [0, 1]"
        n_metrics = computed_metrics.shape[0]
        n_samples = computed_metrics.shape[1]
        weights = np.ones((n_metrics, 1)) if weights == 'ones' else weights
        assert(weights.size == n_metrics and weights.shape[1] == 1)
        return np.prod(computed_metrics * weights, axis=0)
    
    # If more than one metric was used, combine the results (we look for max only after combining to make sure the time lags correspond).
    # The default is to average the cross-correlation/convolution of different metrics at corresponding time lags.
    # A product of the energies commonly appears at the denominator (see e.g. https://en.wikipedia.org/wiki/Coherence_(signal_processing));
    # however, ((x+y)/2)**2 > xy for all x,y > 0: using the average enhances differences in the input energies, 
    # and allows us to discriminate between distinct constant input signals x=X, y=Y (XY / ((X+Y)/2)**2 < 1)
    energy_per_signalA = autocorrA[:, 0].reshape(n_metrics, 1)
    energy_per_signalB = autocorrB[:, 0].reshape(n_metrics, 1)
    crosscorr_metrics = np.sign(crosscorr) * crosscorr**2 / (energy_per_signalA * energy_per_signalB)
    crosscorr_metrics = np.clip(crosscorr_metrics, a_min=0.0, a_max=1.0)  # negative scores are clipped to 0
    ensemble_crosscorr = combine_metrics(crosscorr_metrics)
    if convol is not None:
        convol_metrics = np.clip(np.sign(convol) * convol**2 / (energy_per_signalA * energy_per_signalB), a_min=0.0, a_max=1.0)  # negative scores are clipped to 0
        ensemble_convol = combine_metrics(convol_metrics)
        
    # Max between convolution and correlation guarantees that the similarity metric 
    # is agnostic to the orientation of the parametrization
    if convol is not None:
        sim = max(np.max(ensemble_crosscorr), np.max(ensemble_convol))
    else:
        sim = np.max(ensemble_crosscorr)
    
    return sim

def curve_distance_shift(rod1,rod2,metrics=['curvature', 'curv-times-tors'], discretization_type=CurvatureDiscretizationType.Angle,mirror = False, reverse = False):
    rod_list = [rod1,rod2]
    autocorr, crosscorr, convol = compute_correlation_and_convolution(rod_list, metrics=metrics)
    
    if mirror:
        if metrics == ['curvature', 'curv-times-tors']:# or metrics == ['curvature'] or metrics == ['curv-times-tors']: --> the second line expects 2 metrics for now
            crosscorr_mirr = copy(crosscorr)
            crosscorr_mirr[:, :, 1, :] *= -1  # flip curv-times-tors
            if reverse:
                convol_mirr = copy(convol)
                convol_mirr[:, :, 1, :] *= -1
        else:
            raise NotImplementedError('Cluster mirrored not implemented for custom metrics.')
    
    if not mirror:
        if not reverse:
            similarity = compute_similarity_from_correlation(autocorr[0, :, :], autocorr[1, :, :], crosscorr[0, 1, :, :])
        else:
            similarity = compute_similarity_from_correlation(autocorr[0, :, :], autocorr[1, :, :], crosscorr[0, 1, :, :], convol[0, 1, :, :])
    else:
        if not reverse:
            M_orig = compute_similarity_from_correlation(autocorr[0, :, :], autocorr[1, :, :], crosscorr     [0, 1, :, :])
            M_mirr = compute_similarity_from_correlation(autocorr[0, :, :], autocorr[1, :, :], crosscorr_mirr[0, 1, :, :])
            similarity = max(M_orig, M_mirr)
        else:
            M_orig = compute_similarity_from_correlation(autocorr[0, :, :], autocorr[1, :, :], crosscorr     [0, 1, :, :], convol     [0, 1, :, :])
            M_mirr = compute_similarity_from_correlation(autocorr[0, :, :], autocorr[1, :, :], crosscorr_mirr[0, 1, :, :], convol_mirr[0, 1, :, :])
            similarity = max(M_orig, M_mirr)
    similarity = np.clip(similarity, a_min=0.0, a_max=1.0)
    return 1 - similarity