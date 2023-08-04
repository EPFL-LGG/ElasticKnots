import numpy as np
from sklearn.decomposition import PCA
import elastic_rods
from elastic_rods import CurvatureDiscretizationType
import os
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from copy import copy
from elastic_knots import PeriodicRodList

# ------------------------------------------------------------------------
#                         Periodic rods
# ------------------------------------------------------------------------
    
def define_periodic_rod(pts, material, rest_curv_rad=np.inf, total_opening_angle=0, minimize_twist=False):
    duplicated_0 = np.linalg.norm(pts[0, :] - pts[-2, :]) < 1e-12
    duplicated_1 = np.linalg.norm(pts[1, :] - pts[-1, :]) < 1e-12
    if not duplicated_0 and not duplicated_1:
        pts = np.vstack((pts, pts[0, :], pts[1, :]))
    elif duplicated_0 != duplicated_1:
        raise ValueError("Only one of the first two nodes was duplicated.")
        
    pr = elastic_rods.PeriodicRod(pts, zeroRestCurvature=True)  # always set rest curvature to zero, then eventually modify restKappas
    pr.setMaterial(material)
    pr.totalOpeningAngle = total_opening_angle
    
    # Set rest curvature
    if rest_curv_rad != np.inf:
        rest_lengths = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        rest_kappas = compute_rest_kappas(rest_curv_rad=rest_curv_rad, rest_lengths=rest_lengths)
        pr.rod.setRestKappas(rest_kappas)
        
    # Set the bending energy type to match the definition from [Bergou et al. 2010]
    # The bending energy in [Bergou et al. 2008] is technically non-physical.
    pr.rod.bendingEnergyType = elastic_rods.BendingEnergyType.Bergou2010
    
    if minimize_twist:
        # Minimize the twisting energy 
        # (i.e. run an optimization on the \theta variables only, 
        # leaving the ends of the rod free to untwist)
        elastic_knots.minimize_twist(pr)
    
    return pr


def define_periodic_circle(npts, material, r, translation=np.array([0, 0, 0]), x_noise=0, y_noise=0, z_noise=0):
    """
    Regular polygon with `npts` edges.
    """
    t = np.linspace(0, 2 * np.pi, npts, endpoint=False)
    dx = x_noise*np.random.uniform(-1, 1, t.size)
    dy = y_noise*np.random.uniform(-1, 1, t.size)
    dz = z_noise*np.random.uniform(-1, 1, t.size)
    t = np.concatenate([t, t[0:2]])
    dx = np.concatenate([dx, dx[0:2]])
    dy = np.concatenate([dy, dy[0:2]])
    dz = np.concatenate([dz, dz[0:2]])
    pts = np.column_stack([r * np.cos(t) + dx, r * np.sin(t) + dy, dz]) + translation
    return define_periodic_rod(pts, material)


def set_generalized_link(pr, gen_link):
    writhe = pr.writhe()
    mat_frame_turns = gen_link - writhe - pr.totalReferenceTwistAngle() / (2*np.pi)
    pr.totalOpeningAngle = 2*np.pi * mat_frame_turns
    
    # Spread twist along the whole rod (instead of leaving it concentrated at the connection, which can cause extreme coiling)
    elastic_knots.spread_twist_preserving_link(pr)
    
    return pr


def vibrational_analysis(rod_list, problemOptions, optimizerOptions, n_modes=8, fixedVars=[]):
    "Compute the vibrational modes of an elastic rod with self-contacts."
    
    from compute_vibrational_modes import compute_vibrational_modes, MassMatrixType
    n_iter_temp = optimizerOptions.niter
    optimizerOptions.niter = 0

    def callback_with_variational_modes(problem, iteration):
        global modes, lambdas
        print("Computing vibrational modes")
        lambdas, modes = compute_vibrational_modes(problem, fixedVars=fixedVars, n=n_modes, mtype=MassMatrixType.IDENTITY)

    report = elastic_knots.compute_equilibrium(
        rod_list, problemOptions, optimizerOptions,
        fixedVars=fixedVars,
        externalForces=np.zeros(rod_list.numDoF()),
        callback=callback_with_variational_modes
    )
    optimizerOptions.niter = n_iter_temp
    
    if report.gradientNorm[-1] > 2*optimizerOptions.gradTol:
        print("WARNING: high gradient norm detected, vibrational modes are supposed to be computed only at equilibrium states!")
    
    return lambdas, modes


def compute_min_eigenval_straight_rod(pr):
    "Compute the eigenvalue corresponsing to the first eigenmode of a straight *open* rod equivalent to the input periodic rod"
    
    # Build a straight *open* elastic rod
    rl = pr.restLengths()
    cum_rl = np.cumsum(rl)
    nv = pr.numVertices()
    pts = np.zeros((nv+1, 3))
    pts_perturbed = np.zeros((nv+1, 3))
    for i in range(1, nv+1):
        pts[i, 0] = cum_rl[i-1]
        pts_perturbed[i] = pts[i] + 1e-2 * pr.rod.characteristicLength() * np.random.random(3)
    straight_rod = elastic_rods.ElasticRod(pts_perturbed)
    straight_rod.setMaterial(pr.rod.material(0))
    
    # Compute equilibrium
    from bending_validation import suppress_stdout
    fixedVars = [0, 1, 2, 3, 4, 5, straight_rod.thetaOffset()]
    with suppress_stdout(): elastic_rods.compute_equilibrium(straight_rod, fixedVars=fixedVars)
    # elastic_rods.compute_equilibrium(straight_rod, fixedVars=fixedVars)
    
    # Compute vibrational modes
    from compute_vibrational_modes import compute_vibrational_modes
    lambdas, modes = compute_vibrational_modes(straight_rod, fixedVars, n=1)
    min_eigenval = lambdas[0]
    
    return min_eigenval


def bounding_box(data):
    if isinstance(data, elastic_rods.ElasticRod) or isinstance(data, elastic_knots.PeriodicRodList) or isinstance(data, elastic_rods.PeriodicRod):
        pts = np.array(data.deformedPoints())
    elif isinstance(data, np.ndarray):
        pts = data
    bbmin = np.min(pts, axis=0)
    bbmax = np.max(pts, axis=0)
    bbox = np.array([bbmin, bbmax])
    return bbox


def center_periodic_rod(pr):
    dx = -np.mean(np.array(pr.deformedPoints()), axis=0)
    return translate_periodic_rod(pr, dx)


def translate_periodic_rod(pr, dx):
    pts = np.array(pr.rod.deformedPoints())
    pts += dx
    pr.rod.setDeformedConfiguration(pts, pr.rod.thetas())
    return pr


def scale_periodic_rod(pr, scale):
    pts = np.array(pr.rod.deformedPoints())
    pts *= scale
    pr.rod.setDeformedConfiguration(pts, pr.rod.thetas())
    return pr

# ------------------------------------------------------------------------
#                             Curve similarity
# ------------------------------------------------------------------------

def compute_correlation_and_convolution(rod_list, metrics=['curvature', 'curv-times-tors'], discretization_type=CurvatureDiscretizationType.Angle, pointwise=True):
    from scipy import fftpack
    
    def uniform_resample(x, y, n_samples, kind='linear'):
        from scipy import interpolate
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
                _, y = uniform_resample(cum_edge_lengths, rod.curvature(discretization_type, pointwise=pointwise), n_samples, kind=resampling_kind)
                signals_ri.append(y)
            elif metric == 'torsion':
                _, y = uniform_resample(cum_edge_lengths, rod.torsion(discretization_type, pointwise=pointwise), n_samples, kind=resampling_kind)
                signals_ri.append(y)
            elif metric == 'curv-times-tors':
                torsion = rod.torsion(discretization_type, pointwise=pointwise)   # per vertex
                if np.all(np.abs(torsion) < 1e-8):
                    torsion[:] = 1e-10  # torsion can be close to zero everywhere (e.g. circle) => numerical fluctuations can result in low similarity score
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
    crosscorr_metrics = np.clip(np.sign(crosscorr) * crosscorr**2 / (energy_per_signalA * energy_per_signalB), a_min=0.0, a_max=1.0)  # negative scores are clipped to 0
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

    
def compute_similarity_matrix(rod_list, metrics=['curvature', 'curv-times-tors'], discretization_type=CurvatureDiscretizationType.Angle, cluster_reversed=True, cluster_mirrored=True):
    "Compute the pairwise similarity score of a list of closed curves"

    n_rods = len(rod_list)
    autocorr, crosscorr, convol = compute_correlation_and_convolution(rod_list, metrics=metrics)
    
    if not cluster_reversed:
        del convol
    
    if cluster_mirrored:
        if metrics == ['curvature', 'curv-times-tors']:
            crosscorr_mirr = copy(crosscorr)
            crosscorr_mirr[:, :, 1, :] *= -1  # flip curv-times-tors
            if cluster_reversed:
                convol_mirr = copy(convol)
                convol_mirr[:, :, 1, :] *= -1
        else:
            raise NotImplementedError('Cluster mirrored not implemented for custom metrics.')   
    
    M = np.zeros((n_rods, n_rods))
    for a in range(n_rods):
        for b in range(n_rods):
            if b < a:
                continue
            elif a == b:
                M[a, b] = 1.0
                continue
                
            if not cluster_mirrored:
                if not cluster_reversed:
                    M[a, b] = compute_similarity_from_correlation(autocorr[a, :, :], autocorr[b, :, :], crosscorr[a, b, :, :])
                elif cluster_reversed:
                    M[a, b] = compute_similarity_from_correlation(autocorr[a, :, :], autocorr[b, :, :], crosscorr[a, b, :, :], convol[a, b, :, :])

            elif cluster_mirrored:
                if not cluster_reversed:
                    M_orig = compute_similarity_from_correlation(autocorr[a, :, :], autocorr[b, :, :], crosscorr     [a, b, :, :])
                    M_mirr = compute_similarity_from_correlation(autocorr[a, :, :], autocorr[b, :, :], crosscorr_mirr[a, b, :, :])
                    M[a, b] = max(M_orig, M_mirr)
                elif cluster_reversed:
                    M_orig = compute_similarity_from_correlation(autocorr[a, :, :], autocorr[b, :, :], crosscorr     [a, b, :, :], convol     [a, b, :, :])
                    M_mirr = compute_similarity_from_correlation(autocorr[a, :, :], autocorr[b, :, :], crosscorr_mirr[a, b, :, :], convol_mirr[a, b, :, :])
                    M[a, b] = max(M_orig, M_mirr)

            M[a, b] = np.clip(M[a, b], a_min=0.0, a_max=1.0)  # make sure to e.g. clip 1+1e-16 to 1.0
                
    M += np.triu(M, k=1).T  # symmetrize
    return M


def compute_pairwise_similarity(rod_a, rod_b, *args, **kwargs):
    M = compute_similarity_matrix([rod_a, rod_b], *args, **kwargs)
    return M[0, 1]


# ------------------------------------------------------------------------
#                               Clustering
# ------------------------------------------------------------------------

def get_clusters_from_cut_tree(Z, cut_height, sorted_leaves=None):
    from scipy.cluster.hierarchy import linkage, cut_tree
    t = cut_tree(Z, height=cut_height)
    clustering_labels = t.flatten()
    
    if sorted_leaves is not None:
        # Renumber cluster labels to match the order in dendrogram's leaves
        renumbered_clustering_labels = copy(clustering_labels)
        seq_clust = -1
        prev_clust = -1
        curr_clust = -1
        for i, l in enumerate(sorted_leaves):
            curr_clust = clustering_labels[l]
            if curr_clust != prev_clust:
                seq_clust += 1
            renumbered_clustering_labels[l] = seq_clust
            prev_clust = curr_clust
        clustering_labels = renumbered_clustering_labels
    
    eq_clusters = get_clusters_from_labels(clustering_labels)
    
    return eq_clusters


def get_clusters_from_labels(clustering_labels):
    cluster_indices = np.unique(clustering_labels)
    n_knots = len(clustering_labels)
    knot_indices = np.arange(0, n_knots)
    eq_clusters = {}
    for ci in cluster_indices:
        eq_clusters[ci] = knot_indices[clustering_labels == ci]
    return eq_clusters
    
    
def hierarchical_clustering_from_similarity_matrix(M, plot=True, linkage_method='complete', cut_thresh=None, **kwargs):
    from scipy.cluster.hierarchy import linkage, cut_tree
    import scipy.spatial.distance as ssd
    
    if np.all(M == 1.0):  # single cluster
        n = M.shape[0]
        eq_clusters = {0: np.arange(n)}
        Z = np.array([])
        cuts = []
        cut_thresh = 0.0  
        return eq_clusters, Z, cuts, cut_thresh

    M_dist = 1 - M
    M_dist_condensed = ssd.squareform(M_dist)
    Z = linkage(M_dist_condensed, method=linkage_method)

    cuts = Z[Z[:, 2] > 1e-10, 2]  # get (positive) distances at which the dendrogram branches
    cuts = np.append(0.0, cuts)  # add first cut (all atomic clusters)
    cuts = np.append(cuts, 1.01*cuts[-1])   # add last cut (single cluster)
    
    # Compte the dendrogram (even if we do not plot it)
    # to extract the leaves' order and sort the clusters accordingly 
    from scipy.cluster.hierarchy import dendrogram
    if plot:
        fig, ax = plt.subplots(figsize=(8, 4))
        dn = dendrogram(Z, color_threshold=-1, ax=ax)
    else:
        dn = dendrogram(Z, color_threshold=-1, no_plot=True)
    sorted_leaves = dn['leaves']
    
    assert(cut_thresh is not None)
    assert(cut_thresh >= 0.0 and cut_thresh <= 1.0)
    eq_clusters = get_clusters_from_cut_tree(Z, cut_thresh, sorted_leaves)

    if plot:
        ax.hlines(cut_thresh, *ax.get_xlim(), linestyle='--', linewidth=1, color='C1') 
        ax.set_ylabel('Distance = 1 - similarity')
        plt.show()
        
    return eq_clusters, Z, cuts, cut_thresh, dn


def plot_dendrogram_and_clusters(M, cut_thresh):
    from scipy.cluster.hierarchy import cut_tree

    # Compute dendrogram
    eq_clusters, Z, cuts, cut_thresh, dn = hierarchical_clustering_from_similarity_matrix(
        M, plot=False, linkage_method='complete', cut_thresh=cut_thresh,
    )
    
    # Plot dendrogram
    fontsize = 12
    from scipy.cluster.hierarchy import dendrogram
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={'width_ratios': [1.2, 1]})
    _ = dendrogram(Z, color_threshold=-1, no_plot=False, ax=axes[0])    # plot dendrogram
    axes[0].hlines(cut_thresh, *axes[0].get_xlim(), linestyle='--', linewidth=1.5, color='C1') 
    axes[0].set_ylabel('Distance = 1 - Similarity', fontsize=fontsize)
    axes[0].set_ylim([-0.001, 1.0])
    # axes[0].get_xaxis().set_ticks([])
    axes[0].spines[['right', 'top', 'bottom']].set_visible(False)
    axes[0].tick_params(axis='y', labelsize=fontsize)
    
    clustering_labels = cut_tree(Z, height=cut_thresh).flatten()
    import random

    n = M.shape[0]
    cluster_indices = np.unique(clustering_labels)

    matshow_data = axes[1].matshow(M, cmap='Blues')
    cbar = fig.colorbar(matshow_data, ax=axes[1], location='right', shrink=0.8, pad = 0.04)
    cbar_ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    cbar.set_ticks(cbar_ticks)
    cbar.set_ticklabels(cbar_ticks)
    cbar.ax.tick_params(labelsize=fontsize)
    axes[1].set_xlim([-1, n])  # larger limits to accomodate squares drawn on top of the matrix
    axes[1].set_ylim([n, -1])  # flipped axis in matplotlib
    axes[1].set_title('Similarity Matrix')

    sorted_leaves = dn['leaves']
    n_clust = len(cluster_indices)
    for ci in range(n_clust):
        indices_ci = [sorted_leaves[i] for i in np.arange(n)[clustering_labels == ci]]
        max_i_ci = max(indices_ci)
        min_i_ci = min(indices_ci)

        # Plot squares
        m = min_i_ci - 0.4
        L = max_i_ci - min_i_ci + 0.8
        color = 'C1'
        linewidth = 4
        rectangle = plt.Rectangle([m, m], L, L, fc='none', ec=color, linewidth=linewidth)
        axes[1].add_patch(rectangle)

    plt.show()
    
    return cut_thresh


# ------------------------------------------------------------------------
#                            Visualization
# ------------------------------------------------------------------------

def periodic_scalar_field(field, rods, perEdge=True):
    """
    Extend the scalar field for each rod (or for a single `PeriodicRod`).
    Duplicate the first entry (on edges, first two entries on nodes) and append it at the end.
    Needed for visualisation: PeriodicRods are rendered using the underlying ElasticRod.
    """
    if isinstance(rods, elastic_rods.PeriodicRod):
        pr = rods
        if perEdge:
            extendedField = np.array([])
            i, j = 0, pr.rod.numEdges()-1
            extendedField = np.append(extendedField, field[i:j])
            extendedField = np.append(extendedField, field[i])
        else:
            extendedField = np.array([])
            i, j = 0, pr.rod.numVertices()-2
            extendedField = np.append(extendedField, field[i:j])
            extendedField = np.append(extendedField, field[i])
            extendedField = np.append(extendedField, field[i+1])
            
    elif isinstance(rods, elastic_knots.PeriodicRodList):
        if perEdge:
            extendedField = np.array([])
            splitIndices = np.cumsum(np.append(0, rods.numEdgesPerRod()))
            for ri in range(rods.numRods()):
                i, j = splitIndices[ri], splitIndices[ri+1]
                extendedField = np.append(extendedField, field[i:j])
                extendedField = np.append(extendedField, field[i])
        else:
            extendedField = np.array([])
            splitIndices = np.cumsum(np.append(0, rods.numVerticesPerRod()))
            for ri in range(rods.numRods()):
                i, j = splitIndices[ri], splitIndices[ri+1]
                extendedField = np.append(extendedField, field[i:j])
                extendedField = np.append(extendedField, field[i])
                extendedField = np.append(extendedField, field[i+1])
        
    return extendedField


def color_rods(viewer, rods, contacts=[], colors=[], color_metric=None, clip_metric_outliers=False, perEdge=True):
    
    scalar_field = np.zeros(rods.numEdges())
    if len(colors) == 0:
        colors = np.linspace(0.1, 0.9, rods.numRods())
    else:
        assert(len(colors) == rods.numRods())
        
    if color_metric is None:
            
        numEdgesPerRod = rods.numEdgesPerRod()
        if all([type(color) == float for color in colors]):  # one color per rod
            assert(all(np.array(colors) >= 0.0) and all(np.array(colors) <= 1.0))
        elif all([type(color) == np.ndarray for color in colors]):  # a list of colors per rod
            assert(len(set(color.size for color in colors)) == 1) # check all color vectors have same length
        for ri in range(rods.numRods()):
            offset = rods.firstGlobalNodeIndexInRod(ri)
            scalar_field[offset:offset+numEdgesPerRod[ri]] = colors[ri]  # colors[ri] can be float or numpy.ndarray
            
        if len(contacts) > 0:
            for cIdx, contact in enumerate(contacts):
                scalar_field[contact.contactIdxOver()] = .75
                scalar_field[contact.contactIdxUnder()] = .25
        for ri in range(rods.numRods()): # at last, color connection (bounds color scale)
            offset = rods.firstGlobalNodeIndexInRod(ri)
            scalar_field[offset] = 0
            scalar_field[offset+numEdgesPerRod[ri]-1] = 1
        if isinstance(rods, elastic_knots.PeriodicRodList):
            scalar_field = periodic_scalar_field(scalar_field, rods, perEdge=perEdge)
            
    elif color_metric is not None:
        
        numEdgesPerRod = rods.numEdgesPerRod()
        if color_metric == 'curvature':
            perEdge = False
            for ri in range(rods.numRods()):
                offset = rods.firstGlobalNodeIndexInRod(ri)
                scalar_field[offset:offset+numEdgesPerRod[ri]] = rods[ri].curvature(CurvatureDiscretizationType.Angle, pointwise=True)
        elif color_metric == 'torsion':
            perEdge = True
            for ri in range(rods.numRods()):
                offset = rods.firstGlobalNodeIndexInRod(ri)
                scalar_field[offset:offset+numEdgesPerRod[ri]] = rods[ri].torsion(CurvatureDiscretizationType.Angle, pointwise=True)
        elif color_metric == 'elastic_forces':
            perEdge = False
            for ri in range(rods.numRods()):
                offset = rods.firstGlobalNodeIndexInRod(ri)
                forces = -rods[ri].gradient()[0:3*rods[ri].numVertices()].reshape(-1, 3)
                forces_magnitude = np.linalg.norm(forces, axis=1)
                scalar_field[offset:offset+numEdgesPerRod[ri]] = forces_magnitude
        elif color_metric == 'jet':
            perEdge = False
            scalar_field = np.zeros(rods.numEdges())
            for ri in range(rods.numRods()):
                offset = rods.firstGlobalNodeIndexInRod(ri)
                scalar_field[offset:offset+numEdgesPerRod[ri]] = np.linspace(0, 1, rods[ri].numEdges())
                
        if clip_metric_outliers:
            metricMin = np.percentile(scalar_field, 1)
            metricMax = np.percentile(scalar_field, 99)
            scalar_field = np.clip(scalar_field, metricMin, metricMax)  # clamp outliers
        scalar_field = periodic_scalar_field(scalar_field, rods, perEdge=perEdge)

    viewer.update(mesh=rods, preserveExisting=False, scalarField=scalar_field)


def align_point_cloud_principal_components_to_axes(pts, center=True, ax_pc_0=np.array([0, 1, 0]), ax_pc_1=np.array([1, 0, 0])):   
    "Rotate a point cloud to align its first two principal components to a pair of given (orthogonal) vectors"
    
    centroid = np.mean(pts, axis=0)
    pts -= centroid
    
    # Add random rotation to avoid numerical instabilities due to pts being already approximately aligned to PCs
    R_rand = Rotation.random().as_matrix()
    pts = pts @ R_rand
    
    pca = PCA(n_components=3)
    pca.fit(pts)
    V = pca.components_  # principal directions
    Rx = rotation_matrix_from_vectors(V[0, :], ax_pc_0) # align first PC to given axis
    Ry = rotation_matrix_from_vectors(Rx @ V[1, :], ax_pc_1) # align second PC to given axis
    newPts = ((Ry @ Rx) @ pts.T).T
    
    if not center:
        newPts += centroid
        
    return newPts


def compute_grid_step(rods, step_factor=1.0):
    "Compute the grid size to accomodate the visualization of a list of rods without intersections"
    grid_step = 0
    for r in rods:
        bb_size = np.diff(bounding_box(r), axis=0)[0]
        grid_step = max(grid_step, step_factor*max(bb_size))
    return grid_step


def build_clustered_grid(rods, eq_clusters, grid_step=None, step_factor=1):
    if grid_step is None:
        grid_step = compute_grid_step(rods, step_factor)
        
    # Build PeriodicRodList with clustered rods, clustering similiar equilibria together
    newRods = []
    hShift = 0
    for ci in eq_clusters.keys():
        nCurrEqTypes = len(eq_clusters[ci])
        nCurrRows = np.ceil(np.sqrt(nCurrEqTypes))
        iGridCluster = 0
        jGridCluster = hShift
        for gridIndex, ri in enumerate(eq_clusters[ci]):
            translatedRod = copy(rods[ri])
            iGrid = iGridCluster + gridIndex % nCurrRows
            jGrid = jGridCluster + int(gridIndex/nCurrRows)
            translation = np.array([jGrid, -iGrid, 0]) * grid_step
            centerline = translatedRod.deformedPoints() + translation
            translatedRod.setDeformedConfiguration(centerline, translatedRod.thetas())
            newRods.append(translatedRod)
        hShift += int((len(eq_clusters[ci]) - 1)/nCurrRows + 2)
    return PeriodicRodList(newRods)


def build_regular_grid(rods, grid_step=None, grid_step_x=None, grid_step_y=None, n_rows=None, n_cols=None, step_factor=1.0):
    """
    Rectangular grid (each column has same #rows).
    Input: list of PeriodicRods.
    """
    if grid_step is None and grid_step_x is None and grid_step_y is None:
        grid_step = compute_grid_step(rods, step_factor)
        grid_step_x = grid_step
        grid_step_y = grid_step
    elif grid_step is not None:
        assert(grid_step_x is None and grid_step_y is None)
        grid_step_x = grid_step
        grid_step_y = grid_step
    else:
        assert(grid_step_x is not None and grid_step_y is not None)
        
    nRods = len(rods)
    if not ((n_rows is None) ^ (n_cols is None)):
        n_rows = int(np.ceil(np.sqrt(nRods)))

    # Build PeriodicRodList with translated rods
    # Only one between n_rows and n_cols is not None
    if n_rows is not None:
        rodsPerRow = int(np.ceil(nRods / n_rows))
    elif n_cols is not None:
        rodsPerRow = n_cols
        
    newRods = []
    for i, r in enumerate(rods):
        translatedRod = copy(r)
        iGrid = int(i/rodsPerRow)
        jGrid = i % rodsPerRow
        translation = np.array([jGrid*grid_step_x, -iGrid*grid_step_y, 0])
        centerline = translatedRod.deformedPoints() + translation
        translatedRod.setDeformedConfiguration(centerline, translatedRod.thetas())
        newRods.append(translatedRod)
    return PeriodicRodList(newRods)

# ------------------------------------------------------------------------
#                                    I/O
# ------------------------------------------------------------------------
import elastic_knots

def write_obj(file, rod, center=True, separate_files=False):

    path = os.path.dirname(file)
    os.makedirs(path, exist_ok=True)
    
    def write_obj_data_to_file(file, points_list, ne_per_rod, center):
        if center:  # for PeriodicRodList, globally center 
            points_list = list(np.array(points_list) - np.mean(np.array(points_list), axis=0))
        
        with open(file, 'w') as f:
            for p in points_list:
                f.write('v ')
                for coord in p:
                    f.write(str(coord) + ' ')
                f.write('\n')
            cum_ne = np.append(0, np.cumsum(ne_per_rod)[0:-1] + 1)
            for start_ne, ne in zip(cum_ne, ne_per_rod):
                for i in range(start_ne+1, start_ne+ne):  # nodes' numbering starts from 1
                    f.write('l ' + str(i) + ' ' + str(i+1) + '\n')
                f.write('l ' + str(start_ne+ne) + ' ' + str(start_ne+1) + '\n')
    
    if isinstance(rod, elastic_rods.PeriodicRod):
        points_list = rod.deformedPoints()
        write_obj_data_to_file(file, points_list, [rod.numEdges()], center)
        
    elif isinstance(rod, elastic_knots.PeriodicRodList):
        rod_list = rod
        points_list = rod_list.deformedPoints()
        n_rods = rod_list.size()
        nepr = rod_list.numEdgesPerRod()
        if separate_files:
            for ri in range(n_rods):
                fni = rod_list.firstGlobalNodeIndexInRod(ri)
                ne = nepr[ri]
                curr_file = file.split('.obj')[0] + str(ri) + '.obj'
                write_obj_data_to_file(curr_file, points_list[fni:fni+ne], [ne], center)
        else:
            write_obj_data_to_file(file, points_list, nepr, center)
            
    elif isinstance(rod, np.ndarray):  # assume single rod
        points_list = list(rod)
        write_obj_data_to_file(file, points_list, [rod.shape[0]], center)
        
    else:
        raise ValueError('Unknown input type')
        
        
def write_txt(file, rod, center=True):
    
    def write_txt_data_to_file(file, points_list, ne_per_rod, center):
        if center:  # for PeriodicRodList, globally center 
            points_list = list(np.array(points_list) - np.mean(np.array(points_list), axis=0))
        np.savetxt(file, np.array(points_list))

    if isinstance(rod, elastic_rods.PeriodicRod):
        points_list = rod.deformedPoints()
        write_txt_data_to_file(file, points_list, [rod.numEdges()], center)
    elif isinstance(rod, elastic_knots.PeriodicRodList):
        points_list = rod.deformedPoints()
        write_txt_data_to_file(file, points_list, rod.numEdgesPerRod(), center)
    elif isinstance(rod, np.ndarray):  # assume single rod
        points_list = list(rod)
        write_txt_data_to_file(file, points_list, [rod.shape[0]], center)
    else:
        raise ValueError('Unknown input type')
        

def read_nodes_from_file(file):
    """
    Supported extensions: obj, txt
    """
    nodes = []
    connectivity = []
    n_rods = 0
    if file.endswith('.obj'):
        with open(file, 'r') as f:
            for i, line in enumerate(f):
                if line.startswith('v'):
                    pt = []
                    for coord in line.split(' ')[1:4]:
                        pt.append(float(coord))
                    nodes.append(np.array(pt))
                if line.startswith('l'):
                    edge = []
                    for index in line.split(' ')[1:3]:
                        edge.append(int(index))
                        if len(edge) == 2 and int(index) < edge[0]: # last edge of a rod 
                            n_rods += 1
                    connectivity.append(edge)
        if n_rods == 1:
            return np.array(nodes)
        elif n_rods > 1:
            indices_connections = [i for i in range(len(connectivity)) if connectivity[i][0] > connectivity[i][1]]
            ne_per_rod = np.append(indices_connections[0] + 1, np.diff(indices_connections))
            pts = np.array(nodes)
            pts_list = [pts[0:ne_per_rod[0], :]]
            for ri in range(0, n_rods-1):
                pts_list.append(pts[indices_connections[ri]:indices_connections[ri+1]])
            return pts_list
    
    elif file.endswith('.txt'):
        with open(file, 'r') as f:
            for i, line in enumerate(f):
                pt = []
                for coord in line.split(' ')[0:3]:
                    pt.append(float(coord))
                nodes.append(np.array(pt))
        return np.array(nodes)
    
    elif not '.' in file.split('/')[0]: # no extension, assum same formatting as .txt
        with open(file, 'r') as f:
            for i, line in enumerate(f):
                pt = []
                for coord in line.split(' ')[0:3]:
                    pt.append(float(coord))
                nodes.append(np.array(pt))
        return np.array(nodes)
    

def load_rods_from_dataframe(df, data_path, material, n_rows=None, n_cols=None, step_factor=1, grid_step_x=None, grid_step_y=None):
    knot_types = df['knot_type']
    eq_ids = df['eq_id']
    
    rods = []
    for knot_type, eq_id in zip(knot_types, eq_ids):
        file = data_path + knot_type + '/' + str(eq_id).zfill(4) + '.obj'
        pts = read_nodes_from_file(file)
        pts_aligned = align_point_cloud_principal_components_to_axes(pts)
        pr = define_periodic_rod(pts_aligned, material, minimize_twist=True)
        rods.append(pr)
        
    # rodsList = build_grid(rods)
    rodsList = build_regular_grid(rods, n_rows=n_rows, n_cols=n_cols, step_factor=step_factor, grid_step_x=grid_step_x, grid_step_y=grid_step_y)
    return rodsList
# ------------------------------------------------------------------------
#                                 Misc
# ------------------------------------------------------------------------
    
def load_knot_table(path='../data/knotinfo_table.csv'):
    "Parse the KnotInfo knot table (source: https://knotinfo.math.indiana.edu)"
    
    import pandas as pd
    knot_data = pd.read_csv(path)
    knot_data['Braid Notation'] = knot_data['Braid Notation'].apply(lambda x: list(x.split('};{'))[0][1::] + '}' if x.startswith('[') else x) # get only first braid word if multiple are present
    knot_data['Braid Notation'] = knot_data['Braid Notation'].apply(lambda x: list(map(int, x.replace('{', '').replace('}', '').split(';')))) # parse braid word to list
    return knot_data


def sorted_nicely(l): 
    """
    Sort the given iterable in the way that humans expect.
    Source: https://stackoverflow.com/questions/2669059/how-to-sort-alpha-numeric-set-in-python
    
    ['4_1', '11n_10', '11a_4', '11a_1', '3_1', '12a_1022'] -> ['3_1', '4_1', '11a_1', '11a_4', '11n_10', '12a_1022']
    """ 
    import re
    convert = lambda text: int(text) if text.isdigit() else text 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)


def download_data(gdrive_id, output_dir, zip_file, unzip=True, delete_zip=True):
    "Download and unzip data from GDrive"
    import gdown
    import zipfile
    from tqdm import tqdm
        
    target_zip = output_dir + zip_file
    target_dir = target_zip.replace('.zip', '')
    if unzip and os.path.isdir(target_dir):
        print('Directory {} already exists'.format(target_dir))
        return
    if os.path.isfile(target_zip):
        if not unzip:
            print('File {} already exists'.format(target_zip))
            return
        else:
            print('File {} already exists; unzipping...'.format(target_zip))
    else:
        gdown.download(id=gdrive_id, output=target_zip, quiet=False)
    
    if unzip:
        with zipfile.ZipFile(target_zip) as zf:
            for member in tqdm(zf.infolist(), desc='Extracting '):
                try:
                    zf.extract(member, output_dir)
                except zipfile.error as e:
                    pass

    if delete_zip:
        os.remove(target_zip)
        

def rotation_matrix_from_vectors(vec1, vec2):
    """ 
    Source: https://stackoverflow.com/questions/45142959/calculate-rotation-matrix-to-align-two-vectors-in-3d-space
    Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    if np.dot(a, b) > 1 - 1e-8:
        return np.eye(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix