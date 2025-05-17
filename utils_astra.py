import numpy as np
import astra
import matplotlib.pyplot as plt
import os
from skimage.io import imsave

def create_astra_geometry(phantom_shape, num_angles, detector_size=None):
    """
    Create ASTRA geometry.

    Parameters:
        phantom_shape: Shape of the phantom image (rows, cols)
        num_angles: Number of projection angles
        detector_size: Number of detector elements (defaults to image width)

    Returns:
        proj_geom: ASTRA projection geometry
        vol_geom: ASTRA volume geometry
    """
    if detector_size is None:
        detector_size = phantom_shape[1]
    
    # Create volume geometry
    vol_geom = astra.create_vol_geom(phantom_shape[0], phantom_shape[1])

    det_count = int(np.sqrt(2) * phantom_shape[0])
    
    # Create parallel beam projection geometry
    angles = np.linspace(0, np.pi, num_angles, endpoint=False)
    proj_geom = astra.create_proj_geom('parallel', 1.0, det_count, angles)
    
    return proj_geom, vol_geom

def generate_sinogram_astra(phantom, proj_geom, vol_geom):
    """
    Generate sinogram of a phantom using ASTRA.

    Parameters:
        phantom: Input phantom image (2D numpy array)
        proj_geom: ASTRA projection geometry
        vol_geom: ASTRA volume geometry

    Returns:
        sinogram: Generated sinogram (angles × detector positions)
    """
    phantom_id = astra.data2d.create('-vol', vol_geom, phantom)
    sinogram_id = astra.data2d.create('-sino', proj_geom, 0)
    
    cfg = astra.astra_dict('FP')
    cfg['ProjectionDataId'] = sinogram_id
    cfg['VolumeDataId'] = phantom_id
    cfg['ProjectorId'] = astra.create_projector('line', proj_geom, vol_geom)

    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id)

    sinogram = astra.data2d.get(sinogram_id)

    # Clean up
    astra.algorithm.delete(alg_id)
    astra.data2d.delete(phantom_id)
    astra.data2d.delete(sinogram_id)
    
    return sinogram


def SIRT(vol_geom, vol_data, sino_id, iters=2000, use_gpu=False):
    # create starting reconstruction
    rec_id = astra.data2d.create('-vol', vol_geom, data=vol_data)
    # define SIRT config params
    alg_cfg = astra.astra_dict('SIRT_CUDA' if use_gpu else 'SIRT')
    alg_cfg['ProjectionDataId'] = sino_id
    alg_cfg['ReconstructionDataId'] = rec_id
    alg_cfg['option'] = {}
    alg_cfg['option']['MinConstraint'] = 0
    alg_cfg['option']['MaxConstraint'] = 255
    # define algorithm
    alg_id = astra.algorithm.create(alg_cfg)
    # run the algorithm
    astra.algorithm.run(alg_id, iters)
    # create reconstruction data
    rec = astra.data2d.get(rec_id)

    return rec_id, rec
    
def reconstruct_sirt_astra(sinogram, proj_geom, vol_geom, num_iterations=10, mask=None):
    """
    Perform SIRT reconstruction using ASTRA.

    Parameters:
        sinogram: Input sinogram
        proj_geom: ASTRA projection geometry
        vol_geom: ASTRA volume geometry
        num_iterations: Number of iterations
        mask: Optional mask to restrict updates (None means full update)

    Returns:
        Reconstructed image
    """
    sinogram_id = astra.data2d.create('-sino', proj_geom, sinogram)
    recon_id = astra.data2d.create('-vol', vol_geom, 0)

    cfg = astra.astra_dict('SIRT')
    cfg['ReconstructionDataId'] = recon_id
    cfg['ProjectionDataId'] = sinogram_id
    cfg['ProjectorId'] = astra.create_projector('line', proj_geom, vol_geom)
    cfg['option'] = {}
    cfg['option']['MinConstraint'] = 0
    cfg['option']['MaxConstraint'] = 255
    
    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id)

    reconstruction = astra.data2d.get(recon_id)

    # Apply mask if given
    if mask is not None:
        phantom_id = astra.data2d.create('-vol', vol_geom, reconstruction)
        recon_mask_id = astra.data2d.create('-vol', vol_geom, 0)

        cfg_mask = astra.astra_dict('SIRT')
        cfg_mask['ReconstructionDataId'] = recon_mask_id
        cfg_mask['ProjectionDataId'] = sinogram_id
        cfg_mask['ProjectorId'] = cfg['ProjectorId']
        
        cfg_mask['option'] = {'ReconstructionMaskId': phantom_id}

        alg_mask_id = astra.algorithm.create(cfg_mask)
        astra.algorithm.run(alg_mask_id, num_iterations)

        reconstruction = astra.data2d.get(recon_mask_id)
        astra.data2d.delete(recon_mask_id)
        astra.data2d.delete(phantom_id)

    # Clean up
    astra.algorithm.delete(alg_id)
    astra.data2d.delete(sinogram_id)
    astra.data2d.delete(recon_id)

    return reconstruction

def display_sinogram(sinogram):
    """Display the sinogram image."""
    plt.figure(figsize=(10, 5))
    plt.imshow(sinogram, cmap='gray', aspect='auto', 
               extent=[0, sinogram.shape[1], 180, 0])
    plt.xlabel('Detector Position')
    plt.ylabel('Projection Angle (degrees)')
    plt.title('Sinogram')
    plt.colorbar()
    plt.show()

def add_noise(sinogram, noise_type='poisson', noise_level=1.0):
    """
    Add noise to the sinogram.

    Parameters:
        sinogram: Input sinogram
        noise_type: Type of noise ('poisson' or 'gaussian')
        noise_level: Intensity of noise

    Returns:
        Noisy sinogram
    """
    if noise_type == 'poisson':
        max_val = np.max(sinogram)
        scaled_sino = sinogram / max_val * noise_level * 255
        noisy_sino = np.random.poisson(scaled_sino) * max_val / (noise_level * 255)
        return noisy_sino
    elif noise_type == 'gaussian':
        noise = np.random.normal(0, noise_level * np.mean(sinogram), sinogram.shape)
        return sinogram + noise
    else:
        return sinogram

def compute_rmse(original, reconstructed):
    """
    Compute Root Mean Square Error (RMSE) between the original and reconstructed images.
    """
    return np.sqrt(np.mean((original - reconstructed) ** 2))

def compute_rnmp(original, reconstructed):
    """
    Compute relative Normalized Mean Pixel error (rNMP) between the original and reconstructed images.
    """
    return np.linalg.norm(original - reconstructed) / np.linalg.norm(original)

def save_image(image, filepath):
    """
    Save an image to the specified filepath.

    Parameters:
        image: Input image
        filepath: Destination path
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    if image.dtype == np.float32 or image.dtype == np.float64:
        image = (255 * (image - np.min(image)) / (np.max(image) - np.min(image))).astype(np.uint8)
    imsave(filepath, image)

def reconstruct_fbp_astra(sinogram, proj_geom, vol_geom, num_iterations=10):
    """
    Perform Filtered Back Projection (FBP) reconstruction using ASTRA.

    Parameters:
        sinogram: Input sinogram
        proj_geom: ASTRA projection geometry
        vol_geom: ASTRA volume geometry

    Returns:
        Reconstructed image
    """
    sinogram_id = astra.data2d.create('-sino', proj_geom, sinogram)
    recon_id = astra.data2d.create('-vol', vol_geom, 0)
    projector_id = astra.create_projector('line', proj_geom, vol_geom)
    
    astra.algorithm.run(alg_id, num_iterations)
    
    cfg = astra.astra_dict('FBP')
    cfg['ReconstructionDataId'] = recon_id
    cfg['ProjectionDataId'] = sinogram_id
    cfg['ProjectorId'] = projector_id
    cfg['option'] = {'FilterType': 'Ram-Lak'}

    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id)

    reconstruction = astra.data2d.get(recon_id)

    # Clean up
    astra.algorithm.delete(alg_id)
    astra.data2d.delete(sinogram_id)
    astra.data2d.delete(recon_id)
    astra.projector.delete(projector_id)

    return reconstruction

def display_absolute_difference(image1, image2, title="Absolute Difference"):
    """
    Display the absolute difference between two images.

    Parameters:
        image1: First image (numpy array).
        image2: Second image (numpy array).
        title: Title for the plot.
    """
    abs_diff = np.abs(image1 - image2)
    plt.figure(figsize=(6, 6))
    plt.imshow(abs_diff, cmap='hot')
    plt.colorbar(label="Intensity Difference")
    plt.title(title)
    plt.axis("off")
    plt.show()
