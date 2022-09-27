
import numpy as np

def context_feat(feat_mat, left_context, right_context):
    #///OG feat_mat: T * dim
    if left_context == 0 and right_context == 0:
        return feat_mat
    context_mat = [feat_mat]

    #if left_context is set, we create new matrix in which each subseq submatrix contains top row repeated, and then reverse
    for i in range(left_context):
        context_mat.append(np.vstack((context_mat[-1][0], context_mat[-1][:-1])))
    context_mat.reverse()

    #similar to left_context, but here the last row is repeated, and we do not reverse
    for i in range(right_context):
        context_mat.append(np.vstack((context_mat[-1][1:], context_mat[-1][-1])))

    #concat and stack all submatrices
    return np.hstack(context_mat)

def skip_feat(feat_mat, skip):
    if skip == 1 or skip == 0:
        return feat_mat
    skip_feat = []

    #iterate through rows, and skippign if necessary; return vertically stacked of nonskipped rows
    for i in range(feat_mat.shape[0]):
        if i % skip == 0:
            skip_feat.append(feat_mat[i])
    return np.vstack(skip_feat)


