from scipy.spatial.distance import cdist
import numpy as np

def matchDescriptors(query_descriptors, database_descriptors, lambd):
    """% Returns a 1xQ matrix where the i-th coefficient is the index of the
    % database descriptor which matches to the i-th query descriptor.
    % The descriptor vectors are QxM and DxM where M is the descriptor
    % dimension and Q and D the amount of query and database descriptors
    % respectively.

    matches(i) will be zero if there is no database descriptor
    % with an SSD < lambda * min(SSD). No two non-zero elements of matches will
    % be equal."""

    # Determine the threshold
    dists = cdist(query_descriptors, database_descriptors)
    nonzero = np.nonzero(dists)
    valid_dists = dists[nonzero[0], nonzero[1]]
    dist_min = np.min(valid_dists)
    delta = lambd*dist_min

    # Iterate and match
    matches = []
    for query_idx in range(query_descriptors.shape[0]):
        database_dists = dists[query_idx, :]
        min_dist = np.min(database_dists)
        proposed_idx = int(np.argwhere(database_dists == min_dist))

        # TODO: Sort matches by minimum distance
        if min_dist < delta and proposed_idx not in matches:
            matches.append(proposed_idx)
        else:
            matches.append(0)
    return np.asarray(matches)




