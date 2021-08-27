import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
import numpy as np
from copy import deepcopy
import cv2
from sklearn.neighbors import KDTree

def processHeatMap(hm, cmap = plt.get_cmap('jet')):
    resize_transform = T.Compose([T.ToPILImage()])
    hm = torch.Tensor(hm)
    hm = np.uint8(cmap(np.array(hm)) * 255)
    return hm


def postprocess(pred, threshold=100):
    """
    Runs the depth image through the model.
    Returns the dense prediction of corners, outer edges, inner edges, and a three-channel image with all three.
    """
    # pred = np.load('/media/ExtraDrive1/clothfolding/test_data/pred_62_19_01_2020_12:53:16.npy')

    corners = processHeatMap(pred[:, :, 0])
    outer_edges = processHeatMap(pred[:, :, 1])
    inner_edges = processHeatMap(pred[:, :, 2])

    corners = corners[:,:,0]
    corners[corners<threshold] = 0
    corners[corners>=threshold] = 255

    outer_edges = outer_edges[:,:,0]
    outer_edges[outer_edges<threshold] = 0
    outer_edges[outer_edges>=threshold] = 255

    inner_edges = inner_edges[:,:,0]
    inner_edges[inner_edges<threshold] = 0
    inner_edges[inner_edges>=threshold] = 255

    return corners, outer_edges, inner_edges

def select_grasp(pred, surface_normal, num_neighbour=100, grasp_target="edges"):
    corners, outer_edges, inner_edges = postprocess(pred)

    impred = np.zeros((corners.shape[0], corners.shape[1], 3), dtype=np.uint8)
    impred[:, :, 0] += corners
    impred[:, :, 1] += outer_edges
    impred[:, :, 2] += inner_edges

    idxs = np.where(corners == 255)
    corners[:] = 1
    corners[idxs] = 0
    idxs = np.where(outer_edges == 255)
    outer_edges[:] = 1
    outer_edges[idxs] = 0
    idxs = np.where(inner_edges == 255)
    inner_edges[:] = 1
    inner_edges[idxs] = 0

    # Choose pixel in pred to grasp
    channel = 1 if grasp_target == 'edges' else 0
    indices = np.where(impred[:, :, channel] == 255) # outer_edge
    if len(indices[0]) == 0:
        print("ERROR: NO FEATURE FOUND")
        return 0, 0, 0, 0, 0

    """
    Grasp based on confidence
    """
    # Filter out ambiguous points
    # impred:[im_height, im_width, 3] -> corner, outer edge, inner edge predictions
    segmentation = deepcopy(impred)
    im_height, im_width, _ = segmentation.shape
    segmentation[np.logical_and(impred[:,:,1]==255, impred[:,:,2]==255),2] = 0
    segmentation[np.logical_and(impred[:,:,1]==255, impred[:,:,2]==255),1] = 0

    inner_edges_filt = np.ones((im_height, im_width))

    inner_edges_filt[segmentation[:,:,2]==255] = 0

    # Get outer-inner edge correspondence
    xx, yy =  np.meshgrid([x for x in range(im_width)],
                        [y for y in range(im_height)])

    print("segmentation shape", segmentation.shape, "surface_normal shape", surface_normal.shape)
    if grasp_target == 'edges':
        xx_o = xx[segmentation[:,:,1]==255]
        yy_o = yy[segmentation[:,:,1]==255]
        surface_normal_o_z = surface_normal[:, :, 2][segmentation[:,:,1]==255]
    else:
        xx_o = xx[segmentation[:,:,0]==255]
        yy_o = yy[segmentation[:,:,0]==255]
        surface_normal_o_z = surface_normal[:, :, 2][segmentation[:,:,0]==255]

    xx_i = xx[segmentation[:,:,2]==255]
    yy_i = yy[segmentation[:,:,2]==255]

    _, lbl = cv2.distanceTransformWithLabels(inner_edges_filt.astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)

    loc = np.where(inner_edges_filt==0)
    xx_inner = loc[1]
    yy_inner = loc[0]
    label_to_loc = [[0,0]]

    for j in range(len(yy_inner)):
        label_to_loc.append([yy_inner[j],xx_inner[j]])

    label_to_loc = np.array(label_to_loc)
    direction = label_to_loc[lbl]
    # Calculate distance to the closest inner edge point for every pixel in the image
    distance = np.zeros(direction.shape)

    distance[:,:,0] = np.abs(direction[:,:,0]-yy)
    distance[:,:,1] = np.abs(direction[:,:,1]-xx)

    # Normalize distance vectors
    mag = np.linalg.norm([distance[:,:,0],distance[:,:,1]],axis = 0)+0.00001
    distance[:,:,0] = distance[:,:,0]/mag
    distance[:,:,1] = distance[:,:,1]/mag

    # Get distances of outer edges
    distance_o = distance[segmentation[:,:,1]==255,:]

    # Get outer edge neighbors of each outer edge point
    num_neighbour = num_neighbour

    # For every outer edge point, find its closest K neighbours
    if len(xx_o) < num_neighbour:
        print("ERROR: Not enough points found for KD-Tree")
        return 0, 0, 0, 0, 0

    tree = KDTree(np.vstack([xx_o,yy_o]).T, leaf_size=2)
    dist, ind = tree.query(np.vstack([xx_o,yy_o]).T, k=num_neighbour)

    xx_neighbours = distance_o[ind][:,:,1]
    yy_neighbours = distance_o[ind][:,:,0]
    xx_var = np.var(xx_neighbours,axis = 1)
    yy_var = np.var(yy_neighbours,axis = 1)
    var = xx_var+yy_var
    var = (var-np.min(var))/(np.max(var)-np.min(var))

    # calculate the center of edges/corners
    mask = inner_edges | outer_edges | corners
    idxs = np.where(mask == 1)
    x_center, y_center = np.mean(idxs[0]), np.mean(idxs[1])
    print("x_center, y_center", x_center, y_center)


    k_var = 1.0
    k_dist_right = 2.
    k_normal_z = 0.5
    dist_right = -(xx_o - x_center) / pred.shape[0]
    cost = k_var * var + k_dist_right * dist_right + k_normal_z * np.abs(surface_normal_o_z)

    # Choose min var point
    cost_min = np.min(cost)
    min_idxs = np.where(cost == cost_min)[0]
    print("Number of min cost indices: %d" % len(min_idxs))
    idx = np.random.choice(min_idxs)
    x = xx_o[idx]
    y = yy_o[idx]


    # predicting angle
    temp, lbl = cv2.distanceTransformWithLabels(inner_edges.astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    loc = np.where(inner_edges==0)
    xx_inner = loc[1]
    yy_inner = loc[0]
    label_to_loc = list(zip(yy_inner, xx_inner))
    label_to_loc.insert(0, (0, 0)) # 1-indexed
    label_to_loc = np.array(label_to_loc)
    direction = label_to_loc[lbl]
    outer_pt = np.array([y, x])
    inner_pt = direction[y, x]


    v = inner_pt - outer_pt
    magn = np.linalg.norm(v)

    if magn == 0:
        error_msg = "magnitude is zero for %d samples" % retries
        print(error_msg)
        magn = 1.0

    unitv = v / magn
    originv = [0, 1] # [y, x]
    angle = np.arccos(np.dot(unitv, originv))

    if v[0] < 0:
        angle = -angle

    return outer_pt[0], outer_pt[1], angle, inner_pt[0], inner_pt[1]
