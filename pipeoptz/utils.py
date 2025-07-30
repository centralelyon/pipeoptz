import cv2
import numpy as np
import random as rd
from math import comb
from sklearn.cluster import KMeans
from scipy.spatial import ConvexHull
from scipy.ndimage import label, rotate as _rotate

def _height_width_ratio(points, angle_rad):
            rotated = rotate_points(points, angle_rad)
            min_xy = rotated.min(axis=0)
            max_xy = rotated.max(axis=0)
            w, h = max_xy - min_xy
            return h / w if w != 0 else 0.0

def get_pos(el: np.ndarray, bonus: int = 0) -> tuple[int, int, int, int]:
    """
    Calculates the bounding box coordinates of a binary mask.

    Args:
        el (np.ndarray): The input binary mask.
        bonus (int, optional): An optional padding to add around the bounding box. Defaults to 0.

    Returns:
        tuple[int, int, int, int]: A tuple (x1, y1, x2, y2) representing the
                                   top-left and bottom-right coordinates of the bounding box.
    """
    xy = np.argwhere(el) # Find all non-zero points
    y1, x1 = xy[:, 0].min(), xy[:, 1].min() # Min y and x
    y2, x2 = xy[:, 0].max(), xy[:, 1].max() # Max y and x
    # Apply bonus, ensuring coordinates are within image bounds
    return max(x1 - bonus, 0), max(y1 - bonus, 0), min(x2 + bonus, el.shape[1] - 1), min(y2 + bonus, el.shape[0] - 1)

def unzip(points: list[tuple]) -> tuple[list, list]:
    """
    Separates a list of 2D points (tuples) into two lists: one for x-coordinates and one for y-coordinates.

    Args:
        points (list[tuple]): A list of (x, y) coordinates.

    Returns:
        tuple[list, list]: A tuple containing two lists: ([x1, x2,...], [y1, y2,...]).
    """
    return ([el[0] for el in points], [el[1] for el in points])

def generate_line(start: tuple[int, int], stop: tuple[int, int]) -> list[tuple[int, int]]:
    """
    Generates a list of integer coordinate points forming a line between start and stop points.
    Uses a simple line drawing algorithm (similar to Bresenham's principle but simplified).

    Args:
        start (tuple[int, int]): The (x, y) coordinates of the starting point.
        stop (tuple[int, int]): The (x, y) coordinates of the ending point.

    Returns:
        list[tuple[int, int]]: A list of (x, y) integer coordinates representing the line.
    """
    dx = stop[0] - start[0]
    dy = stop[1] - start[1]
    sign_y = 1 if dy > 0 else -1
    sign_x = 1 if dx > 0 else -1

    if dx == 0: # Vertical line
        return [(start[0], start[1] + i * sign_y) for i in range(abs(dy) + 1)]
    elif dy == 0: # Horizontal line
        return [(start[0] + i * sign_x, start[1]) for i in range(abs(dx) + 1)]
    elif abs(dy) > abs(dx): # More vertical than horizontal
        return [(start[0] + int(i * dx / dy * sign_x / sign_y), start[1] + i * sign_y) for i in range(abs(dy) + 1)]
    else: # More horizontal than vertical
        return [(start[0] + i * sign_x, start[1] + int(i * dy / dx * sign_y / sign_x)) for i in range(abs(dx) + 1)]

def distance(pt: tuple[float, float]) -> float:
    """
    Calculates the squared Euclidean distance of a 2D point from the origin (0,0).

    Args:
        pt (tuple[float, float]): The (x, y) coordinates of the point.

    Returns:
        float: The squared distance (x^2 + y^2).
    """
    return (pt[0]**2 + pt[1]**2)**0.5

def product(*iterables, random=False, max_combinations=0, optimize_memory=False):
    """
    Returns the cartesian product of input iterables, with an option for random sampling.

    Args:
        *iterables: Variable number of iterables to compute the product.
        random (bool): If True, returns a random sample from the product instead of all combinations.
        max_combinations (int): The maximum number of combinations to sample.
        optimize_memory (bool): Have an effect only if random is True and max_combinations > 0. 
            If True, optimizes memory usage by generating a random product
            without storing all combinations in memory. But  there is a risk of generating the same 
            value multiple times. Put to True only if max_combinations << len(all_combinations) or if there is no problem
            if the same value is repeated.

    Yields:
        Tuples representing the cartesian product of the input iterables.
    """
    len_index = [len(iterable) for iterable in iterables]
    max_combinations = max_combinations if max_combinations > 0 else np.prod(len_index)

    if random and optimize_memory:
        for i in range(max_combinations):
            yield tuple(it[rd.randrange(length)] for it, length in zip(iterables, len_index))
        return
    
    from itertools import product as it_product
    if random:
        rd_index = list(it_product(*[range(length) for length in len_index]))
        rd.shuffle(rd_index)
        for i in range(min(max_combinations, len(rd_index))):
            yield tuple(iterables[j][rd_index[i][j]] for j in range(len(iterables)))
        return
    
    prod = it_product(*iterables)
    for i in range(min(max_combinations, np.prod(len_index))):
        yield next(prod)

def ith_subset(n: int, i: int) -> list[int]:
    """
    Returns the i-th subset of A = [0, n-1], ordered first by cardinality,
    then lexicographically within each cardinality class.

    Args:
        n (int): Upper bound of the interval A = [0, n-1].
        i (int): Index (0 <= i < 2^n) in the cardinality-sorted power set.

    Returns:
        list[int]: The i-th subset under the cardinality-lex order.
    """
    total = 2**n
    if i < 0 or i >= total:
        raise ValueError(f"Index i must be in [0, {total - 1}]")

    # Find the cardinality group (number of elements in subset)
    remaining = i
    for k in range(n + 1):  # cardinalities from 0 to n
        c = comb(n, k)
        if remaining < c:
            cardinality = k
            break
        remaining -= c

    # Generate the `remaining`-th k-combination in lex order
    subset = []
    x = 0
    for j in range(cardinality):
        while comb(n-1 - x, cardinality - j - 1) <= remaining:
            remaining -= comb(n-1 - x, cardinality - j - 1)
            x += 1
        subset.append(x)
        x += 1
    return subset

def mse_loss(image_a, image_b):
    """Calculate the mean squared error (MSE) between two images."""
    if image_a is None or image_b is None:
        return float('inf')
    err = np.sum((image_a.astype("float") - image_b.astype("float")) ** 2)
    err /= float(image_a.shape[0] * image_a.shape[1])
    return err

def remove_color(image: np.ndarray, color: tuple[int, int, int], tolerance: int) -> np.ndarray:
    """
    Removes a specific color from an image by making pixels of that color transparent.
    Assumes the input image has an alpha channel (RGBA). If not, it might behave unexpectedly
    or should be adapted (e.g., by setting to black or white).
    This version sets matching pixels to (0,0,0,0) assuming RGBA.

    Args:
        image (np.ndarray): The input RGBA image (H, W, 4).
        color (tuple[int, int, int]): The RGB color to remove.
        tolerance (int): The tolerance for color matching (Euclidean distance in RGB space).

    Returns:
        np.ndarray: The image with the specified color removed (made transparent).
    """
    if image.shape[2] < 3:
        raise ValueError("Image must have at least 3 channels (RGB).")
    
    new_im = image.copy()
    # Calculate squared Euclidean distance for efficiency
    # (r_img - r_c)^2 + (g_img - g_c)^2 + (b_img - b_c)^2 <= tolerance^2
    color_diff_sq = (image[:,:,0].astype(np.int32) - color[0])**2 + \
                    (image[:,:,1].astype(np.int32) - color[1])**2 + \
                    (image[:,:,2].astype(np.int32) - color[2])**2
    
    mask_to_remove = color_diff_sq <= tolerance**2

    if image.shape[2] == 4: # RGBA image
        new_im[mask_to_remove] = [0, 0, 0, 0] # Set to transparent black
    else: # RGB image, set to black or another placeholder
        new_im[mask_to_remove] = [0, 0, 0]
    return new_im

def remove_colors(image: np.ndarray, colors: list[tuple[int, int, int]], tolerances: list[int]) -> np.ndarray:
    """
    Removes multiple specified colors from an image.

    Args:
        image (np.ndarray): The input RGBA image (H, W, 4).
        colors (list[tuple[int, int, int]]): A list of RGB colors to remove.
        tolerances (list[int]): A list of tolerances, one for each color.

    Returns:
        np.ndarray: The image with the specified colors removed.
    """
    if len(colors) != len(tolerances):
        raise ValueError("Length of colors and tolerances lists must match.")
    
    new_im = image.copy()
    for color, tolerance in zip(colors, tolerances):
        new_im = remove_color(new_im, color, tolerance) # Iteratively remove each color
    return new_im

def isolate(binary_mask: np.ndarray, sizemin: int = 1) -> list[np.ndarray]:
    """
    Isolates connected components (elements) in a binary mask.
    Uses a Breadth-First Search (BFS) or Depth-First Search (DFS) like approach.

    Args:
        binary_mask (np.ndarray): A 2D boolean or integer mask.
        sizemin (int, optional): The minimum size (number of pixels) for an element to be kept.
                                 Defaults to 1.

    Returns:
        list[np.ndarray]: A list of boolean masks, each representing an isolated element.
    """
    if not np.any(binary_mask):
        return []

    # Use scipy.label for a standard and efficient connected components labeling
    labeled_array, num_features = label(binary_mask)
    
    elements = []
    for i in range(1, num_features + 1): # Iterate through each found component
        component_mask = (labeled_array == i)
        if np.sum(component_mask) >= sizemin:
            elements.append(component_mask)
    return elements

def remove_palette(image: np.ndarray, recolored_image: np.ndarray,
                   palette: np.ndarray, indices_to_remove: list[int]) -> np.ndarray:
    """
    Makes regions in the original image transparent if their corresponding color
    in the recolored image matches one of the specified palette colors.
    Assumes original_image is RGBA.

    Args:
        original_image (np.ndarray): The original RGBA image (H, W, 4).
        recolored_image (np.ndarray): An image (H, W, 3 or 4) where pixels are assigned colors from the palette.
        palette (np.ndarray): The color palette (N_colors, 3).
        indices_to_remove (list[int]): A list of indices in the palette. Colors at these indices
                                       will be targeted for removal.

    Returns:
        np.ndarray: The original image with targeted regions made transparent.
    """
    if image.shape[2] != 4:
        raise ValueError("Original image must be RGBA.")

    output_image = image.copy()
    recolored_rgb = recolored_image[:,:,:3]

    for i in indices_to_remove:
        if 0 <= i < len(palette):
            color_to_match = palette[i]
            # Create a mask where the recolored image matches the palette color
            match_mask = np.all(recolored_rgb == color_to_match, axis=2)
            output_image[match_mask, 3] = 0 # Set alpha to 0 (transparent)
    return output_image

def extract_palette(image: np.ndarray, n_colors: int, sample_size: int = 0,
                    max_iter: int = 300, use_lab: bool = False) -> np.ndarray:
    """
    Extracts a color palette from an image using K-Means clustering.
    Considers only non-transparent pixels if the image is RGBA.

    Args:
        image (np.ndarray): The input image (H, W, 3 or 4).
        n_colors (int): The number of colors to extract for the palette.
        sample_size (int, optional): Number of pixels to sample for K-Means.
                                     0 means use all (valid) pixels. Defaults to 0.
        max_iter (int, optional): Maximum iterations for K-Means. Defaults to 300.
        use_lab (bool, optional): If True, perform clustering in LAB color space. Defaults to False.

    Returns:
        np.ndarray: A palette of shape (n_colors, 3) as uint8 RGB values.
    """
    if image.shape[2] == 4: # RGBA
        # Consider only opaque pixels
        opaque_mask = image[:,:,3] != 0
        pixels = image[opaque_mask][:,:3] # Get RGB values of opaque pixels
        if pixels.shape[0] == 0: return np.array([], dtype=np.uint8).reshape(0,3) # No opaque pixels
    elif image.shape[2] == 3: # RGB
        pixels = image.reshape(-1, 3)
    else:
        raise ValueError("Image must be RGB or RGBA.")

    if use_lab:
        pixels_lab = cv2.cvtColor(pixels.reshape(1, -1, 3).astype(np.uint8), cv2.COLOR_RGB2LAB)[0]
        data_for_kmeans = pixels_lab
    else:
        data_for_kmeans = pixels.astype(np.float32) # Kmeans expects float

    if sample_size > 0 and data_for_kmeans.shape[0] > sample_size:
        indices = np.random.choice(data_for_kmeans.shape[0], size=sample_size, replace=False)
        sample = data_for_kmeans[indices]
    else:
        sample = data_for_kmeans
    
    if sample.shape[0] < n_colors: # Not enough samples for desired clusters
        # print(f"Warning: Not enough distinct pixel samples ({sample.shape[0]}) for {n_colors} clusters. Reducing n_colors.")
        n_colors = max(1, sample.shape[0]) # Ensure at least 1 cluster if samples exist
        if n_colors == 0: return np.array([], dtype=np.uint8).reshape(0,3)


    kmeans = KMeans(n_clusters=n_colors, max_iter=max_iter, n_init='auto', random_state=0)
    kmeans.fit(sample)
    centers = kmeans.cluster_centers_

    if use_lab:
        # Sort by L channel (lightness) in LAB space
        # centers_lab_sorted = sorted(centers, key=lambda el: el[0], reverse=True) # L is 0-100
        # Convert LAB centers back to RGB
        # OpenCV LAB: L in [0,255], A,B in [0,255] (representing ~[-128,127])
        # Standard LAB: L in [0,100], A,B in ~[-128,127]
        # Assuming centers are in OpenCV's LAB range if cvtColor was used
        palette_lab = np.array(centers, dtype=np.float32).reshape(1, -1, 3)
        palette_rgb = cv2.cvtColor(palette_lab, cv2.COLOR_LAB2RGB)[0]
        palette = np.clip(palette_rgb, 0, 255).astype(np.uint8)
        # Optional: sort RGB palette by luminance or other criteria if needed after conversion
        # For now, use LAB sorting order
        # Re-sort based on L value of the original LAB centers for consistency
        l_values = centers[:, 0]
        sorted_indices = np.argsort(l_values)[::-1] # Descending L
        palette = palette[sorted_indices]

    else: # RGB space
        # Sort by intensity (sum of squares)
        # palette = sorted(centers, key=lambda el: (el**2).sum(), reverse=True)
        # palette = np.clip(palette, 0, 255).astype(np.uint8)
        # No, Kmeans centers are already means, so they are the colors.
        # Sorting by luminance (approx)
        luminance = 0.299 * centers[:, 0] + 0.587 * centers[:, 1] + 0.114 * centers[:, 2]
        sorted_indices = np.argsort(luminance)[::-1] # Descending luminance
        palette = np.clip(centers[sorted_indices], 0, 255).astype(np.uint8)

    return palette

def recolor(image: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """
    Recolors an image using a given palette. Each pixel in the original image
    is replaced by the closest color from the palette.
    Handles RGB and RGBA images. For RGBA, alpha channel is preserved.

    Args:
        image (np.ndarray): The input image (H, W, 3 or 4).
        palette (np.ndarray): The color palette (N_colors, 3) as uint8 RGB.

    Returns:
        np.ndarray: The recolored image with the same dimensions and type as input.
    """
    h, w, c = image.shape
    is_rgba = (c == 4)
    
    rgb_image_part = image[:, :, :3].astype(np.float32)
    pixels_flat = rgb_image_part.reshape(-1, 3) # (H*W, 3)
    palette_float = palette.astype(np.float32) # (N_colors, 3)

    # Calculate distances from each pixel to each palette color
    # dists will be (H*W, N_colors)
    dists = np.linalg.norm(pixels_flat[:, np.newaxis, :] - palette_float[np.newaxis, :, :], axis=2)
    
    # Find the index of the closest palette color for each pixel
    nearest_palette_indices = np.argmin(dists, axis=1) # (H*W,)
    
    # Create the new image using these palette colors
    recolored_rgb_flat = palette[nearest_palette_indices] # (H*W, 3)
    recolored_rgb = recolored_rgb_flat.reshape(h, w, 3).astype(np.uint8)

    if is_rgba:
        alpha_channel = image[:, :, 3:] # Keep original alpha
        recolored_image = np.dstack((recolored_rgb, alpha_channel))
    else:
        recolored_image = recolored_rgb
        
    return recolored_image

def min_size(image: np.ndarray) -> np.ndarray:
    """
    Crops an RGBA image to the bounding box of its non-transparent content.

    Args:
        image (np.ndarray): The input RGBA image (H, W, 4).

    Returns:
        np.ndarray: The cropped RGBA image. Returns an empty array if image is fully transparent.
    """
    if image.shape[2] != 4:
        raise ValueError("Input image must be RGBA.")

    alpha_channel = image[:, :, 3]
    if not np.any(alpha_channel): # Fully transparent
        return np.array([], dtype=image.dtype).reshape(0,0,4)

    rows_any = np.any(alpha_channel, axis=1)
    cols_any = np.any(alpha_channel, axis=0)

    row_min, row_max = np.where(rows_any)[0][[0, -1]]
    col_min, col_max = np.where(cols_any)[0][[0, -1]]

    return image[row_min:row_max+1, col_min:col_max+1, :]

def remove_alpha(image: np.ndarray, background_color: tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    """
    Converts an RGBA image to RGB by replacing transparent pixels with a specified background color.

    Args:
        image (np.ndarray): The input RGBA image (H, W, 4).
        background_color (tuple[int, int, int], optional): The RGB color to use for transparent areas.
                                                           Defaults to white (255, 255, 255).

    Returns:
        np.ndarray: The RGB image (H, W, 3).
    """
    if image.shape[2] != 4:
        raise ValueError("Input image must be RGBA.")

    rgb_part = image[:, :, :3]
    alpha_part = image[:, :, 3]

    # Create a mask for transparent pixels
    transparent_mask = (alpha_part == 0)

    image_rgb = rgb_part.copy()
    image_rgb[transparent_mask] = background_color
    return image_rgb.astype(np.uint8)

def slic(image: np.ndarray, num_superpixels: int = 100, compactness: float = 20.0, max_iter: int = 10) -> np.ndarray:
    """
    Segments an image into superpixels using a simplified SLIC algorithm.
    Note: For robust SLIC, consider using skimage.segmentation.slic.
    This is a basic implementation.

    Args:
        image (np.ndarray): The input RGB image (H, W, 3), uint8.
        num_superpixels (int, optional): The desired number of superpixels. Defaults to 100.
        compactness (float, optional): Balances color proximity and space proximity. Higher values make
                                     superpixels more square. Defaults to 20.0.
        max_iter (int, optional): Maximum number of iterations. Defaults to 10.

    Returns:
        np.ndarray: A 2D array of labels (H, W), where each unique label corresponds to a superpixel.
    """
    h, w = image.shape[:2]
    lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)

    # Approximate side length of a superpixel
    S = int(np.sqrt(h * w / num_superpixels))
    if S == 0: S = 1 # Avoid S=0 for very small images or large num_superpixels

    # Initialize cluster centers on a grid
    centers_list = []
    for y_grid in range(S // 2, h, S):
        for x_grid in range(S // 2, w, S):
            l, a, b = lab_image[y_grid, x_grid]
            centers_list.append([l, a, b, float(y_grid), float(x_grid)]) # l, a, b, y, x
    
    if not centers_list: # No centers could be initialized
        return np.zeros((h,w), dtype=np.int32)
        
    centers = np.array(centers_list, dtype=np.float32)
    num_actual_centers = centers.shape[0]

    labels = -np.ones((h, w), dtype=np.int32)
    distances = np.full((h, w), np.inf, dtype=np.float32)

    # Create coordinate grids for spatial distance calculation
    yy, xx = np.mgrid[0:h, 0:w]

    for iteration in range(max_iter):
        for i in range(num_actual_centers):
            l_c, a_c, b_c, y_c, x_c = centers[i]

            # Search in a 2S x 2S window around the center
            y_start, y_end = max(0, int(y_c - S)), min(h, int(y_c + S + 1))
            x_start, x_end = max(0, int(x_c - S)), min(w, int(x_c + S + 1))

            if y_start >= y_end or x_start >= x_end: continue # Window is empty

            window_lab = lab_image[y_start:y_end, x_start:x_end]
            window_yy = yy[y_start:y_end, x_start:x_end]
            window_xx = xx[y_start:y_end, x_start:x_end]

            # Color difference (squared Euclidean in LAB)
            color_diff_sq = (window_lab[:,:,0] - l_c)**2 + \
                            (window_lab[:,:,1] - a_c)**2 + \
                            (window_lab[:,:,2] - b_c)**2
            
            # Spatial difference (squared Euclidean in xy)
            spatial_diff_sq = (window_yy - y_c)**2 + (window_xx - x_c)**2

            # Combined distance (SLIC distance measure)
            # D = sqrt(d_lab^2 + (m/S)^2 * d_xy^2)
            # Using squared distances to avoid sqrt until the end if possible,
            # but comparison needs consistent metric. Original SLIC uses sqrt.
            combined_dist = np.sqrt(color_diff_sq + (compactness / S)**2 * spatial_diff_sq)

            # Update labels and distances
            current_distances_in_window = distances[y_start:y_end, x_start:x_end]
            update_mask = combined_dist < current_distances_in_window
            
            current_distances_in_window[update_mask] = combined_dist[update_mask]
            labels[y_start:y_end, x_start:x_end][update_mask] = i
        
        # Update centers to the mean of their assigned pixels
        new_centers = np.zeros_like(centers)
        counts = np.zeros(num_actual_centers, dtype=int)
        for label_idx in range(num_actual_centers):
            mask_for_label = (labels == label_idx)
            if np.any(mask_for_label):
                points_lab = lab_image[mask_for_label]
                points_yy = yy[mask_for_label]
                points_xx = xx[mask_for_label]
                
                new_centers[label_idx, 0] = np.mean(points_lab[:,0])
                new_centers[label_idx, 1] = np.mean(points_lab[:,1])
                new_centers[label_idx, 2] = np.mean(points_lab[:,2])
                new_centers[label_idx, 3] = np.mean(points_yy)
                new_centers[label_idx, 4] = np.mean(points_xx)
                counts[label_idx] = points_lab.shape[0]
            else: # Handle case where a center has no assigned pixels
                new_centers[label_idx] = centers[label_idx] # Keep old center

        if np.allclose(centers, new_centers, atol=1e-2) and iteration > 0: # Check for convergence
            break 
        centers = new_centers

    return labels

def get_angle_min_area(filter_mask: np.ndarray, method='convex_hull', optimize="area") -> float:
    """
    Finds the optimal angle to rotate a binary mask such that its bounding box has
    either the minimum area or the maximum height-to-width ratio.

    Args:
        filter_mask (np.ndarray): The input binary mask (2D boolean or 0/1 array).
        method (str, optional): The optimization method. 'convex_hull' uses the angles
                                of the convex hull edges. 'iterative' iteratively
                                searches for the best angle. Defaults to 'convex_hull'.
        optimize (str, optional): The optimization criterion. 'area' for minimum bounding
                                  box area, 'height_width_ratio' for maximum height-to-width ratio.
                                  Defaults to "area".

    Returns:
        float: The optimal rotation angle in radians.
    """
    if method == 'convex_hull':
        coords = np.argwhere(filter_mask > 0)[:, [1, 0]] # Get (x, y) coordinates of non-zero pixels
        if coords.shape[0] < 3: # ConvexHull requires at least 3 points
            return 0.0

        hull_indices = ConvexHull(coords).vertices
        hull = coords[hull_indices] # Coordinates of the convex hull vertices

        edges = np.roll(hull, -1, axis=0) - hull # Vectors representing the edges of the hull

        # Calculate angles of these edges
        angles = -np.arctan2(edges[:, 1], edges[:, 0]) # Negative for clockwise rotation alignment

        # Create rotation matrices for each angle
        cos_angles = np.cos(angles)
        sin_angles = np.sin(angles)

        rotation_matrices = np.array([
            [cos_angles, -sin_angles],
            [sin_angles,  cos_angles]
        ]).transpose(2, 0, 1)

        projected_coords = np.einsum('ij,ajk->aik', hull, rotation_matrices)

        # Calculate widths and heights of bounding boxes for each rotation
        min_coords = projected_coords.min(axis=1) # (num_angles, 2)
        max_coords = projected_coords.max(axis=1) # (num_angles, 2)
        
        widths = max_coords[:, 0] - min_coords[:, 0]
        heights = max_coords[:, 1] - min_coords[:, 1]
        if optimize == "area":
            areas = widths * heights
            best_index = np.argmin(areas)
        elif optimize == "height_width_ratio":
            ratios = heights / widths
            best_index = np.argmax(ratios)
        else:
            raise ValueError("Invalid optimization criterion. Must be 'area' or 'height_width_ratio'.")
        return angles[best_index]
    
    elif method == 'iterative':
        initial_angle = 0.0
        step = np.pi/8
        precision = 0.01
        
        points = np.vstack(np.where(filter_mask)).T.astype(np.float16)[:, [1,0]] # Get (x,y) coordinates
        if points.shape[0] < 2:
            return initial_angle

        current_angle = initial_angle % np.pi
        if optimize == "height_width_ratio":
            best_ratio = _height_width_ratio(points, current_angle)
        elif optimize == "area":
            rotated = rotate_points(points, new_angle)
            min_xy = rotated.min(axis=0)
            max_xy = rotated.max(axis=0)
            w, h = max_xy - min_xy
            best_ratio = 1/(w*h) if w != 0 else float("inf")
        else:
            raise ValueError("Invalid optimization criterion. Must be 'area' or 'height_width_ratio'.")

        current_step = step
        while current_step > precision:
            improved = False
            for direction in [-1, 1]: # Check angles slightly less and slightly more
                new_angle = (current_angle + direction * current_step) % np.pi
                if optimize == "height_width_ratio":
                    ratio = _height_width_ratio(points, new_angle)
                elif optimize == "area":
                    rotated = rotate_points(points, new_angle)
                    min_xy = rotated.min(axis=0)
                    max_xy = rotated.max(axis=0)
                    w, h = max_xy - min_xy
                    ratio = 1/(w*h) if w != 0 else float("inf")

                if ratio > best_ratio:
                    current_angle = new_angle
                    best_ratio = ratio
                    improved = True
                    break # Found a better angle, restart with this new angle
            if not improved: # If no improvement in either direction, reduce step size
                current_step /= 2
        return current_angle
    else:
        raise ValueError("Invalid method. Must be 'convex_hull' or 'iterative'.")

def rotate_points(points: np.ndarray, angle_rad: float) -> np.ndarray:
    """
    Rotates a set of 2D points around the origin (0,0).

    Args:
        points (np.ndarray): An array of shape (N, 2) representing N points (x, y).
        angle_rad (float): The rotation angle in radians.

    Returns:
        np.ndarray: The array of rotated points, shape (N, 2).
    """
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    rotation_matrix = np.array([[c, -s], [s, c]])
    return points @ rotation_matrix.T # Apply rotation: P' = P * R^T

def rotate(image: np.ndarray, angle_rad: float, order: int = 0, reshape: bool = True) -> np.ndarray:
    """
    Rotates an image by a given angle.

    Args:
        image (np.ndarray): The input image (2D or 3D).
        angle_rad (float): The rotation angle in radians.
        order (int, optional): The order of interpolation (0-5).
                               0: Nearest-neighbor, 1: Bilinear, 3: Bicubic. Defaults to 0.
        reshape (bool, optional): If True, the output shape is adapted to contain the whole
                                  rotated image. If False, the output shape is the same as input.
                                  Defaults to True.

    Returns:
        np.ndarray: The rotated image.
    """
    return _rotate(image, angle_rad * 180 / np.pi, reshape=reshape, order=order)

def find_line(image: np.ndarray, min_threshold: int = 10, max_threshold: int = 50,
              lines_min: int = 3, lines_max: int = 10) -> tuple[tuple[float, float], int] | tuple[None, int]:
    """
    Detects a dominant line in an image using Hough Line Transform with adaptive thresholding.

    Args:
        image (np.ndarray): The input RGB image.
        min_threshold (int, optional): The minimum Hough threshold to start search. Defaults to 10.
        max_threshold (int, optional): The maximum Hough threshold to end search. Defaults to 50.
        lines_min (int, optional): The minimum number of detected lines to consider valid. Defaults to 3.
        lines_max (int, optional): The maximum number of detected lines to consider valid. Defaults to 10.

    Returns:
        tuple[tuple[float, float], int] | tuple[None, int]:
            - A tuple ((rho, theta), threshold) representing the average line parameters (rho, theta)
              and the threshold at which it was found.
            - (None, threshold) if no suitable line is found within the threshold range.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    blur = cv2.GaussianBlur(gray, (5, 5), 1.8)
    # Automatically determine Canny thresholds based on median pixel intensity
    v = np.median(blur[blur != 0]) if np.any(blur != 0) else 0
    sigma = 0.33 # Standard deviation factor for Canny
    lower_canny = int(max(0, (1.0 - sigma) * v))
    upper_canny = int(min(255, (1.0 + sigma) * v))
    edges = cv2.Canny(blur, lower_canny, upper_canny)

    current_threshold = (min_threshold + max_threshold) // 2
    lines = cv2.HoughLines(edges, 1, np.pi / 180 / 20, threshold=current_threshold) # Angle resolution: 1/20th of a degree

    # Iteratively adjust threshold to find a suitable number of lines
    while lines is None or not (lines_min <= len(lines) <= lines_max):
        if lines is not None and len(lines) > lines_max: # Too many lines, increase threshold
            min_threshold = current_threshold + 1
        else: # Too few or no lines, decrease threshold
            max_threshold = current_threshold - 1

        if min_threshold > max_threshold: # Search range exhausted
            # Try one last time with a potentially better threshold if range collapsed
            final_try_threshold = max_threshold +1 if lines is None or (lines is not None and len(lines) < lines_min) else min_threshold -1
            final_try_threshold = max(1, final_try_threshold) # Ensure positive threshold
            lines = cv2.HoughLines(edges, 1, np.pi / 180 / 20, threshold=final_try_threshold)
            return (np.mean(lines, axis=0)[0] if lines is not None else None), final_try_threshold

        current_threshold = (min_threshold + max_threshold) // 2
        if current_threshold <=0: # Prevent non-positive threshold
            return (np.mean(lines, axis=0)[0] if lines is not None else None), 1 # Return current best or none
        lines = cv2.HoughLines(edges, 1, np.pi / 180 / 20, threshold=current_threshold)

    return (np.mean(lines, axis=0)[0] if lines is not None else None), current_threshold

def find_circle(image: np.ndarray, min_radius_denominator: int = 4,
                param1: int = 100, param2: int = 30, blur_ksize: int = 0) -> np.ndarray | None:
    """
    Detects circles in an image using Hough Circle Transform.

    Args:
        image (np.ndarray): The input RGB image.
        min_radius_denominator (int, optional): Denominator to calculate minRadius
                                                (min(height, width) // denominator). Defaults to 4.
        param1 (int, optional): First Canny edge detection threshold for HoughCircles. Defaults to 100.
        param2 (int, optional): Accumulator threshold for circle detection in HoughCircles. Defaults to 30.
        blur_ksize (int, optional): Kernel size for Gaussian blur (if > 0, must be odd).
                                    Defaults to 0 (no blur).

    Returns:
        np.ndarray | None: A NumPy array of detected circles [[x, y, radius], ...], or None if no circles are found.
    """
    im_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2GRAY)

    if blur_ksize > 0 and blur_ksize % 2 == 1: # Apply blur if ksize is positive and odd
        gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0) # Sigma 0 means it's computed from ksize

    min_dim = min(gray.shape[:2])
    min_r = min_dim // min_radius_denominator if min_radius_denominator > 0 else 0
    max_r = min_dim # Max radius can be up to the smallest dimension of the image

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,  # Inverse ratio of accumulator resolution to image resolution
        minDist=max(100, min_r), # Minimum distance between centers of detected circles
        param1=param1, # Higher threshold for Canny edge detector
        param2=param2, # Accumulator threshold for circle centers at detection stage
        minRadius=min_r,
        maxRadius=max_r
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        return circles[0, :] # Return the array of (x, y, radius)
    return None
