import numpy as np
import json

from elastic_knots import *
from elastic_rods import PeriodicRod, RodMaterial, ElasticRod


def load_from_json(filename):
    """ Load tencer to json file.
    Args:
        filename (string) : file name of the tencer's json file
    """
    
    with open(filename) as f:
        tencer_data = json.load(f)
    
    num_closed_rods = tencer_data['numClosedRods']
    num_springs = tencer_data['numSprings']

    young_modulus = tencer_data['YoungModulus']
    c = tencer_data['crossSection']

    closed_rods = []
    springs = []
    attachment_vertices = []
        
    num_rod_defovars = 0 
    for i in range(num_closed_rods):
        rod_points = tencer_data['restPoints'][i]
        closed_rods.append(PeriodicRod(rod_points, zeroRestCurvature=True))
        rod_young_modulus = young_modulus[i]
        material = RodMaterial('ellipse', rod_young_modulus, 0.5, [c[i], c[i]])  # circular cross-section
        closed_rods[-1].setMaterial(material)
        num_rod_defovars += closed_rods[-1].numDoF()
        # closed_rods[-1].rod.setRestLengths(tencer_data['rodRestLengths'][i])
        # closed_rods[-1].rod.setRestDirectors(buildRestDirectors(tencer_data['restDirectors'][i]))
        # closed_rods[-1].rod.deformedConfiguration().initialize_from_data(rod_points,buildRestDirectors(tencer_data['referenceDirectors'][i]),tencer_data['sourceTangents'][i])
        
      
    
    for i in range(num_springs):
        spring_coords = tencer_data['springCoords'][i]
        springs.append(Spring(spring_coords,tencer_data['springStiffnesses'][i],tencer_data['springRestLengths'][i],CompressionType.NoCompression))
        attachment_vertices.append(SpringAttachments(*tencer_data['attachmentVertices'][i]))
        
    tencer = ContactTencer(closed_rods,springs,attachment_vertices)
    
    v = tencer.getDefoVars()
    v[:num_rod_defovars] = tencer_data['defoVars'][:num_rod_defovars]
    tencer.setDefoVars(v)
    
    return tencer

def save_to_json(tencer,filename):
    """ Save a tencer's state in a json file
    Args:
        tencer (Tencer) : a tencer
        filename (string) : the file's name
    """

    closed_rods = tencer.getClosedRods()
    springs = tencer.getSprings()

    dico = dict()
    dico['numClosedRods'] = closed_rods.numRods()
    dico['numSprings'] = len(springs)

    # Rod material
    dico['YoungModulus'] = [1e6 for r in closed_rods] #[r.rod.material(0).youngModulus for r in closed_rods]
    dico['crossSection'] = [r.rod.material(0).crossSectionHeight/2 for r in closed_rods]

    # Rod rest quantities
    dico['rodRestLengths'] = [r.rod.restLengths() for r in closed_rods]
    dico['restPoints'] = [to_list(r.rod.restPoints()) for r in closed_rods]
    # dico['restDirectors'] = [directors_to_list(r.rod.restDirectors()) for r in closed_rods]

    # Rod deformed configuration
    dico['defoVars'] = list(tencer.getDefoVars())
    dico['sourceTangents'] = [to_list(r.rod.deformedConfiguration().sourceTangent) for r in closed_rods]
    dico['referenceDirectors'] = [directors_to_list(r.rod.deformedConfiguration().sourceReferenceDirectors) for r in closed_rods]



    # Springs
    dico['springNumPoints'] = [s.getNumPoints() for s in springs]
    dico['attachmentVertices'] = attachment_vertices_to_list(tencer.getAttachmentVertices())
    dico['springRestLengths'] = [s.getRestLength() for s in springs]
    dico['springStiffnesses'] = [s.getStiffness() for s in springs]
    dico['springCoords'] = [[list(c) for c in s.getCoords()] for s in springs]

    

    with open(filename, 'w') as f:
        json.dump(dico,f,indent=4)




    
def buildRestDirectors(d):
    directors = []
    for i in range(len(d)):
        directors.append(ElasticRod.Directors(d[i][0],d[i][1]))
    return directors

def to_list(l):
    lst = []
    for i in range(len(l)):
        lst.append(list(l[i]))
    return lst

def directors_to_list(d):
    lst = []
    for i in range(len(d)):
        lst.append([list(d[i].d1),list(d[i].d2)])
    return lst

def attachment_vertices_to_list(attachment_vertices):
    lst = []
    for j,v in enumerate(attachment_vertices): 
        rod_idx_list = []
        rod_v_list = []
        spring_v_list = []
        for i in range(len(v.rodIdx)):
            rod_idx_list.append(int(v.rodIdx[i]))
            rod_v_list.append(int(v.rod_vertices[i]))
            spring_v_list.append(int(v.spring_vertices[i]))
        lst.append([list(rod_idx_list),list(rod_v_list),list(spring_v_list)])
    return lst