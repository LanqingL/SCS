import math
import torch
import numpy as np
from sklearn.cluster import KMeans
import torch.nn.functional as F

def cluster_to_fix(can_features, device, args):

    cluster_kmeans = KMeans(n_clusters=args.cluster_num, random_state=0, max_iter=20, n_init='auto')
    kmeans_ret = cluster_kmeans.fit(can_features)

    cluster_centers = torch.from_numpy(cluster_kmeans.cluster_centers_).to(device)
    cluster_id = kmeans_ret.labels_

    return cluster_id, cluster_centers


def cluster_nearest_farthest(cluster_centrid, cluster_features):
    cluster_features = torch.from_numpy(np.stack(cluster_features, axis=0))
    scores = torch.matmul(cluster_features, cluster_centrid.unsqueeze(1))
    # cos_sim = F.cosine_similarity(sub_cluster_features[0], centers, dim=0)

    cand_prob = scores.squeeze(-1).clone().detach()
    cand_prob = cand_prob.cpu().numpy()
    cand_prob = np.nan_to_num(cand_prob, nan=0.000001)  # replace np.nan with 0
    cand_prob /= cand_prob.sum()  # make probabilities sum to 1
    # cids = np.random.choice(range(len(cand_prob)), args.candidate_num, p=cand_prob, replace=False)
    all_cids = sorted(range(len(cand_prob)), key=lambda i: cand_prob[i], reverse=True)
    sample_id = [all_cids[0], all_cids[-1]]
    return sample_id




