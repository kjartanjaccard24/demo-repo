from gs import gs_postprocess
import numpy as np


def get_lcfs(plasma):
    return plasma.ts_fitter.contour_gen(1-plasma.psibar)

def get_lcfs_params(plasma):
    lcfs = get_lcfs(plasma)
    lcfs_params = gs_postprocess.calc_geometric_params(lcfs)
    return lcfs_params

def get_boundary_params(plasma):
    boundary = plasma.ts_fitter.get_boundary(closed=True)
    boundary_params = gs_postprocess.calc_geometric_params(boundary)
    return boundary_params

def get_lowside_curvature(plasma):
    '''
    Calculate the curvature of the low-field side of the LCFS in rad/m.
    Negative values are concave (i.e. bad curvature).
    '''
    lcfs = get_lcfs(plasma)
    if not gs_postprocess.ContourGenerator.is_ccw(lcfs):
        lcfs = lcfs[::-1, :]
    top_idx = np.argmax(lcfs[:, 1])
    bottom_idx = np.argmin(lcfs[:, 1])
    if bottom_idx > top_idx:
        slc = np.r_[bottom_idx:len(lcfs), 0:top_idx+1]
    else:
        slc = np.r_[bottom_idx:top_idx+1]
    lcfs_lowside = lcfs[slc]

    segments = np.diff(lcfs_lowside, axis=0)
    norm = np.linalg.norm(segments, axis=1)
    segments /= norm[:, None]
    cross = segments[:-1, 0]*segments[1:, 1] - segments[:-1, 1]*segments[1:, 0]
    curvature = np.arcsin(cross) * 2/(norm[1:] + norm[:-1])  # rad/m, negative is concave
    return lcfs_lowside, curvature