import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import elastic_rods
from elastic_rods import CurvatureDiscretizationType
import os
import pandas as pd
from scipy.spatial.transform import Rotation

# ------------------------------------------------------------------------
#                       PeriodicRod helpers
# ------------------------------------------------------------------------
    
def define_periodic_rod(pts, material, rest_curv_rad=np.inf, total_opening_angle=0):
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
        restLengths = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        restKappas = compute_rest_kappas(rest_curv_rad=rest_curv_rad, restLengths=restLengths)
        pr.rod.setRestKappas(restKappas)
        
    # Set the bending energy type to match the definition from [Bergou et al. 2010]
    # The bending energy in [Bergou et al. 2008] is technically non-physical.
    pr.rod.bendingEnergyType = elastic_rods.BendingEnergyType.Bergou2010

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
    num_turns = gen_link - writhe - pr.totalReferenceTwistAngle() / (2*np.pi)
    pr.totalOpeningAngle = 2*np.pi * num_turns
    
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


# --------------------------------------------
#              Visualization
# --------------------------------------------


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
    

# --------------------------------------------
#                   I/O
# --------------------------------------------
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
    
    
def load_knot_table(path='../data/knotinfo_table.csv'):
    knots_data = pd.read_csv(path)
    knots_data['Braid Notation'] = knots_data['Braid Notation'].apply(lambda x: list(x.split('};{'))[0][1::] + '}' if x.startswith('[') else x) # get only first braid word if multiple are present
    knots_data['Braid Notation'] = knots_data['Braid Notation'].apply(lambda x: list(map(int, x.replace('{', '').replace('}', '').split(';')))) # parse braid word to list
    return knots_data