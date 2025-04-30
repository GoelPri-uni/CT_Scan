import numpy as np
import astra
import matplotlib.pyplot as plt
import os
from skimage.io import imsave

def create_astra_geometry(phantom_shape, num_angles, detector_size=None):
    """
    创建ASTRA几何结构
    
    参数:
        phantom_shape: phantom图像形状(rows, cols)
        num_angles: 投影角度数量
        detector_size: 探测器单元数量(默认与图像宽度相同)
    
    返回:
        proj_geom: ASTRA投影几何
        vol_geom: ASTRA体积几何
    """
    if detector_size is None:
        detector_size = phantom_shape[1]
    
    # 创建体积几何
    vol_geom = astra.create_vol_geom(phantom_shape[0], phantom_shape[1])
    
    # 创建平行光束投影几何
    angles = np.linspace(0, np.pi, num_angles, endpoint=False)
    proj_geom = astra.create_proj_geom('parallel', 1.0, detector_size, angles)
    
    # 返回投影几何和体积几何
    return proj_geom, vol_geom

def generate_sinogram_astra(phantom, proj_geom, vol_geom):
    """
    使用ASTRA生成phantom的sinogram
    
    参数:
        phantom: 输入的phantom图像(2D numpy数组)
        proj_geom: ASTRA投影几何
        vol_geom: ASTRA体积几何
    
    返回:
        sinogram: 生成的sinogram(角度×探测器位置)
        projector_id: ASTRA投影器ID(需要后续清理)
    """
    # 创建ASTRA需要的volume数据
    phantom_id = astra.data2d.create('-vol', vol_geom, phantom)  # Use vol_geom here
    
    # 创建sinogram存储空间
    sinogram_id = astra.data2d.create('-sino', proj_geom, 0)  # Corrected to handle single return value
    
    # 配置投影算子
    cfg = astra.astra_dict('FP')
    cfg['ProjectionDataId'] = sinogram_id
    cfg['VolumeDataId'] = phantom_id
    cfg['ProjectorId'] = astra.create_projector('line', proj_geom, vol_geom)  # Use vol_geom here
    
    # 创建并运行投影算法
    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id)
    
    # 获取sinogram数据
    sinogram = astra.data2d.get(sinogram_id)
    
    # 清理ASTRA对象
    astra.algorithm.delete(alg_id)
    astra.data2d.delete(phantom_id)
    astra.data2d.delete(sinogram_id)
    
    return sinogram

def reconstruct_sirt_astra(sinogram, proj_geom, vol_geom, num_iterations=10, mask=None):
    """
    使用ASTRA进行SIRT重建
    
    参数:
        sinogram: 输入sinogram
        proj_geom: ASTRA投影几何
        vol_geom: ASTRA体积几何
        num_iterations: 迭代次数
        mask: 只更新mask指定的区域(None表示更新全部)
    
    返回:
        重建图像
    """
    # 创建sinogram数据对象
    sinogram_id = astra.data2d.create('-sino', proj_geom, sinogram)
    
    # 创建重建图像存储空间
    recon_id = astra.data2d.create('-vol', vol_geom, 0)
    
    # 配置SIRT算法
    cfg = astra.astra_dict('SIRT')
    cfg['ReconstructionDataId'] = recon_id
    cfg['ProjectionDataId'] = sinogram_id
    cfg['ProjectorId'] = astra.create_projector('line', proj_geom, vol_geom)
    
    # 创建并运行SIRT算法
    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id, num_iterations)
    
    # 获取重建图像
    reconstruction = astra.data2d.get(recon_id)
    
    # 应用mask
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
    
    # 清理ASTRA对象
    astra.algorithm.delete(alg_id)
    astra.data2d.delete(sinogram_id)
    astra.data2d.delete(recon_id)
    
    return reconstruction

def display_sinogram(sinogram):
    """显示sinogram图像"""
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
    向sinogram添加噪声
    
    参数:
        sinogram: 输入sinogram
        noise_type: 噪声类型('poisson'或'gaussian')
        noise_level: 噪声强度
    
    返回:
        带噪声的sinogram
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
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    if image.dtype == np.float32 or image.dtype == np.float64:
        # Normalize image to 0-255 and convert to uint8
        image = (255 * (image - np.min(image)) / (np.max(image) - np.min(image))).astype(np.uint8)
    imsave(filepath, image)

def reconstruct_fbp_astra(sinogram, proj_geom, vol_geom):
    """
    Perform FBP reconstruction using ASTRA.
    """
    sinogram_id = astra.data2d.create('-sino', proj_geom, sinogram)
    recon_id = astra.data2d.create('-vol', vol_geom, 0)
    projector_id = astra.create_projector('line', proj_geom, vol_geom)  # Add projector ID
    cfg = astra.astra_dict('FBP')
    cfg['ReconstructionDataId'] = recon_id
    cfg['ProjectionDataId'] = sinogram_id
    cfg['ProjectorId'] = projector_id  # Include projector ID
    cfg['option'] = {'FilterType': 'Ram-Lak'}
    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id)
    reconstruction = astra.data2d.get(recon_id)
    astra.algorithm.delete(alg_id)
    astra.data2d.delete(sinogram_id)
    astra.data2d.delete(recon_id)
    astra.projector.delete(projector_id)  # Clean up projector
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