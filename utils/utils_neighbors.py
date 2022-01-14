import numpy as np
from foolbox.distances import LpDistance


def generate_neighboring_points(
    data_sample, amount, epsilon=None, noise_vectors=None, scale=0.1
):
    """
    This function is given a data point data_sample and it generates amount many data points in an epsilon neighborhood
    scale is given to the noise function to sample the random noise
    Source: https://github.com/fraboeni/membership-risk/blob/bdb8190f67588fcf5d79593d2d9cb8307761f17b/code_base/utils.py#L42
    :param data_sample: the data point for which we want to generate neighbors
    :param amount: how many neighbors to generate
    :param epsilon: how far the neighbors should be away in an L2 norm, if not specified, no further clipping is performed
    :param noise_vectors: if we want deterministic noise vectors for neighbor creation, we can pass them here
    :param scale: the sigma for the normal noise distribution
    :return: an array of the generated neighbors all in range [0,1]
    """
    l2_dist = LpDistance(2)
    shape = (amount,) + data_sample.shape  # make a shape of [amount, ax1, ax2, ax3]

    data_sample = np.expand_dims(data_sample, 0)  # reshape for broadcast operations

    # generate noisy samples
    if (
        noise_vectors is not None
    ):  # in case we do need deterministic neighbors they have to be passed
        noises = noise_vectors
    else:
        noises = np.random.normal(loc=0, scale=scale, size=shape)

    # now make neighbors
    perturbed_samples = noises + data_sample  # add data to the noise
    perturbed_samples_clip = np.clip(perturbed_samples, 0, 1)  # bring them to range 0,1

    # and make sure that perturbation does not exceed epsilon
    # the clip function cannot deal with np broadcasts
    if epsilon is not None:
        repeated_data = np.repeat(
            data_sample, amount, axis=0
        )  # therefore, repeat sample

        # the following line is for debugging purpose only:
        a = l2_dist(repeated_data, perturbed_samples_clip)

        perturbed_samples_clip = l2_dist.clip_perturbation(
            repeated_data, perturbed_samples_clip, epsilon
        )

        b = l2_dist(repeated_data, perturbed_samples_clip)
        c = np.isclose(
            b, epsilon
        )  # it's floats so check for every element if roughly the same as epsilon
        # if one is not true, the neighbors might be at the wrong distance
        assert np.all(c), (
            "Your perturbed samples do not seem to have the distance to the original ones that you specified with epsilon. "
            "This might be the case if scale is small and epsilon large"
        )
    return perturbed_samples_clip
