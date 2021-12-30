import numpy as np
from copy import deepcopy

BRIGHTNESS = 50  # default value to adjust brightness pictures


def brighten_images(images: np.ndarray, brightness: int = BRIGHTNESS) -> np.ndarray:
    """
    Adjust the brightness of all input images
    source: https://github.com/privML/privacy-evaluator/blob/main/privacy_evaluator/utils/data_adaptation.py

    :params images: The original images of shape [H, W, D].
    :params brightness: The amount the brightness should be raised or lowered
    :return: Images with adjusted brightness
    """
    brighten_images = deepcopy(images)
    for image in brighten_images:
        _brighten_image(image, brightness)
    return brighten_images


def _brighten_image(image: np.ndarray, brightness: int):
    """
    Adjust the brightness of one image
    :params image: The original image of shape [H, W, D].
    :params brightness: The amount the brightness should be raised or lowered
    """
    height, width, _ = image.shape
    for x in range(height):
        for y in range(width):
            if image[x, y] < (0 - brightness):
                image[x, y] = 0
            elif image[x, y] > (255 - brightness):
                image[x, y] = 255
            else:
                image[x, y] = image[x, y] + brightness
