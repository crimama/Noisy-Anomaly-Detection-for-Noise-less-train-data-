import numpy as np 
from skimage import measure
from scipy.ndimage.measurements import label
from sklearn.metrics import roc_auc_score, auc
from statistics import mean

# def compute_pro(masks: np.ndarray, amaps: np.ndarray, num_th: int = 200) -> None:

#     """Compute the area under the curve of per-region overlaping (PRO) and 0 to 0.3 FPR
#     Args:
#         category (str): Category of product
#         masks (ndarray): All binary masks in test. masks.shape -> (num_test_data, h, w)
#         amaps (ndarray): All anomaly maps in test. amaps.shape -> (num_test_data, h, w)
#         num_th (int, optional): Number of thresholds
#     """

#     assert isinstance(amaps, np.ndarray), "type(amaps) must be ndarray"
#     assert isinstance(masks, np.ndarray), "type(masks) must be ndarray"
#     assert amaps.ndim == 3, "amaps.ndim must be 3 (num_test_data, h, w)"
#     assert masks.ndim == 3, "masks.ndim must be 3 (num_test_data, h, w)"
#     assert amaps.shape == masks.shape, "amaps.shape and masks.shape must be same"
#     #assert set(masks.flatten()) == {0, 1}, "set(masks.flatten()) must be {0, 1}"
#     assert isinstance(num_th, int), "type(num_th) must be int"

#     df = pd.DataFrame([], columns=["pro", "fpr", "threshold"])
#     binary_amaps = np.zeros_like(amaps, dtype=np.bool)

#     min_th = amaps.min()
#     max_th = amaps.max()
#     delta = (max_th - min_th) / num_th

#     for i,th in enumerate(np.arange(min_th, max_th, delta)):
#         binary_amaps[amaps <= th] = 0
#         binary_amaps[amaps > th] = 1

#         pros = []
#         for binary_amap, mask in zip(binary_amaps, masks):
#             for region in measure.regionprops(measure.label(mask)):
#                 axes0_ids = region.coords[:, 0]
#                 axes1_ids = region.coords[:, 1]
#                 tp_pixels = binary_amap[axes0_ids, axes1_ids].sum()
#                 pros.append(tp_pixels / region.area)

#         inverse_masks = 1 - masks
#         fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
#         fpr = fp_pixels / inverse_masks.sum()

        
#         df.loc[i,:] = [mean(pros),fpr,th]
#     # Normalize FPR from 0 ~ 1 to 0 ~ 0.3
#     df = df[df["fpr"] < 0.3]
#     df["fpr"] = df["fpr"] / df["fpr"].max()

#     pro_auc = auc(df["fpr"], df["pro"])
#     return pro_auc        



def compute_pro(anomaly_maps, ground_truth_maps):
    """Compute the PRO curve for a set of anomaly maps with corresponding ground
    truth maps.

    Args:
        anomaly_maps: List of anomaly maps (2D numpy arrays) that contain a
          real-valued anomaly score at each pixel.

        ground_truth_maps: List of ground truth maps (2D numpy arrays) that
          contain binary-valued ground truth labels for each pixel.
          0 indicates that a pixel is anomaly-free.
          1 indicates that a pixel contains an anomaly.

    Returns:
        fprs: numpy array of false positive rates.
        pros: numpy array of corresponding PRO values.
    """

    print("Compute PRO curve...")

    # Structuring element for computing connected components.
    structure = np.ones((3, 3), dtype=int)

    num_ok_pixels = 0
    num_gt_regions = 0

    shape = (len(anomaly_maps),
             anomaly_maps[0].shape[0],
             anomaly_maps[0].shape[1])
    fp_changes = np.zeros(shape, dtype=np.uint32)
    assert shape[0] * shape[1] * shape[2] < np.iinfo(fp_changes.dtype).max, \
        'Potential overflow when using np.cumsum(), consider using np.uint64.'

    pro_changes = np.zeros(shape, dtype=np.float64)

    for gt_ind, gt_map in enumerate(ground_truth_maps):

        # Compute the connected components in the ground truth map.
        labeled, n_components = label(gt_map, structure)
        num_gt_regions += n_components

        # Compute the mask that gives us all ok pixels.
        ok_mask = labeled == 0
        num_ok_pixels_in_map = np.sum(ok_mask)
        num_ok_pixels += num_ok_pixels_in_map

        # Compute by how much the FPR changes when each anomaly score is
        # added to the set of positives.
        # fp_change needs to be normalized later when we know the final value
        # of num_ok_pixels -> right now it is only the change in the number of
        # false positives
        fp_change = np.zeros_like(gt_map, dtype=fp_changes.dtype)
        fp_change[ok_mask] = 1

        # Compute by how much the PRO changes when each anomaly score is
        # added to the set of positives.
        # pro_change needs to be normalized later when we know the final value
        # of num_gt_regions.
        pro_change = np.zeros_like(gt_map, dtype=np.float64)
        for k in range(n_components):
            region_mask = labeled == (k + 1)
            region_size = np.sum(region_mask)
            pro_change[region_mask] = 1. / region_size

        fp_changes[gt_ind, :, :] = fp_change
        pro_changes[gt_ind, :, :] = pro_change

    # Flatten the numpy arrays before sorting.
    anomaly_scores_flat = np.array(anomaly_maps).ravel()
    fp_changes_flat = fp_changes.ravel()
    pro_changes_flat = pro_changes.ravel()

    # Sort all anomaly scores.
    print(f"Sort {len(anomaly_scores_flat)} anomaly scores...")
    sort_idxs = np.argsort(anomaly_scores_flat).astype(np.uint32)[::-1]

    # Info: np.take(a, ind, out=a) followed by b=a instead of
    # b=a[ind] showed to be more memory efficient.
    np.take(anomaly_scores_flat, sort_idxs, out=anomaly_scores_flat)
    anomaly_scores_sorted = anomaly_scores_flat
    np.take(fp_changes_flat, sort_idxs, out=fp_changes_flat)
    fp_changes_sorted = fp_changes_flat
    np.take(pro_changes_flat, sort_idxs, out=pro_changes_flat)
    pro_changes_sorted = pro_changes_flat

    del sort_idxs

    # Get the (FPR, PRO) curve values.
    np.cumsum(fp_changes_sorted, out=fp_changes_sorted)
    fp_changes_sorted = fp_changes_sorted.astype(np.float32, copy=False)
    np.divide(fp_changes_sorted, num_ok_pixels, out=fp_changes_sorted)
    fprs = fp_changes_sorted

    np.cumsum(pro_changes_sorted, out=pro_changes_sorted)
    np.divide(pro_changes_sorted, num_gt_regions, out=pro_changes_sorted)
    pros = pro_changes_sorted

    # Merge (FPR, PRO) points that occur together at the same threshold.
    # For those points, only the final (FPR, PRO) point should be kept.
    # That is because that point is the one that takes all changes
    # to the FPR and the PRO at the respective threshold into account.
    # -> keep_mask is True if the subsequent score is different from the
    # score at the respective position.
    # anomaly_scores_sorted = [7, 4, 4, 4, 3, 1, 1]
    # ->          keep_mask = [T, F, F, T, T, F]
    keep_mask = np.append(np.diff(anomaly_scores_sorted) != 0, np.True_)
    del anomaly_scores_sorted

    fprs = fprs[keep_mask]
    pros = pros[keep_mask]
    del keep_mask

    # To mitigate the adding up of numerical errors during the np.cumsum calls,
    # make sure that the curve ends at (1, 1) and does not contain values > 1.
    np.clip(fprs, a_min=None, a_max=1., out=fprs)
    np.clip(pros, a_min=None, a_max=1., out=pros)

    # Make the fprs and pros start at 0 and end at 1.
    zero = np.array([0.])
    one = np.array([1.])

    return np.concatenate((zero, fprs, one)), np.concatenate((zero, pros, one))