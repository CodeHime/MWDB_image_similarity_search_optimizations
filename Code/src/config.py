

output_dim=(8,8)
hog_dict={
    "orientation_bins": 9,
    "cell_size": 8,
    "block_size": 2,
    "orientated_gradients": False, # ? ASK explaination of this parameter
    "l2_thres": 0.2,
    "norm_type": 'L2-Hys'       # normalization corressponding to max 0.2 threshold
}

distances = ["euclidean", "cosine", "manhattan","earth_movers"]
distance_dict = {
    "img": distances[2],
    "mean": distances[2],
    "std": distances[2],
    "skew": distances[2],
    "elbp": distances[2],
    "hog": distances[1],
    "cm8x8":distances[2],
    "all":distances[2]
}
