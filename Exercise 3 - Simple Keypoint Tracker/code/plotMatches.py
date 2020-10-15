from matplotlib import collections as mc
import matplotlib.pyplot as plt

def plotMatches(matches, query_keypoints, database_keypoints):

    lines = []
    for query_idx, match_idx in enumerate(matches):
        if match_idx != 0:
            query_point = query_keypoints[query_idx, :]
            match_point = database_keypoints[match_idx, :]

            lines.append([(query_point[1], query_point[0]),
                          (match_point[1], match_point[0])])
            lc = mc.LineCollection(lines)
            ax = plt.gca()
            ax.add_collection(lc)
            ax.autoscale()
