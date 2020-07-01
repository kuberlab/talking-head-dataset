import cv2


def image_resize(
        image,
        width=None, height=None,
        enlarge=False, contain_proportions=True,
        inter=cv2.INTER_AREA,
):
    (h, w) = image.shape[:2]
    if not contain_proportions:
        if not width or not height:
            raise RuntimeError("width and height must be set "
                               "if proportions not contained")
        resized = cv2.resize(image, (int(width), int(height)), interpolation=inter)
        aspects = (width / float(w), height / float(h))
        return resized, aspects
    if width is None and height is None:
        return image, 1.
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    elif height is None:
        r = width / float(w)
        dim = (width, int(h * r))
    else:
        rh = height / float(h)
        rw = width / float(w)
        r = min(rh, rw)
        dim = (int(w * r), int(h * r))
    if not enlarge and r > 1:
        return image, 1.
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized, r
