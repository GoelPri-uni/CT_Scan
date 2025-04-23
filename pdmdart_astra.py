import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import minimize
import astra
from utils_astra import reconstruct_sirt_astra

class PDMDARTAstra:
    def __init__(self, sinogram, phantom_shape, num_grey_levels=2):
        """
        初始化PDM-DART重建器(使用ASTRA)
        
        参数:
            sinogram: 输入sinogram数据
            phantom_shape: 重建图像形状
            num_grey_levels: 灰度级数量(默认2:二值图像)
        """
        self.sinogram = sinogram
        self.phantom_shape = phantom_shape
        self.num_grey_levels = num_grey_levels
        self.num_angles, self.detector_size = sinogram.shape
        
        # 初始化参数
        self.grey_levels = np.linspace(0, 1, num_grey_levels)
        self.thresholds = np.linspace(0.3, 0.7, num_grey_levels-1)
        
        # 初始化重建图像
        self.reconstruction = np.zeros(phantom_shape)
        
        # 创建ASTRA几何结构
        self.proj_geom, self.vol_geom = self._create_astra_geometry()
        
        # 创建投影器
        self.proj_id = astra.create_projector('line', self.proj_geom, self.vol_geom)
    
    def _create_astra_geometry(self):
        """创建ASTRA几何结构"""
        return create_astra_geometry(self.phantom_shape, self.num_angles, self.detector_size)
    
    def forward_project(self, image):
        """使用ASTRA进行前向投影"""
        # 创建volume数据对象
        volume_id = astra.data2d.create('-vol', self.vol_geom, image)
        
        # 创建sinogram存储空间
        sinogram_id, sinogram = astra.data2d.create('-sino', self.proj_geom, 0)
        
        # 配置投影算子
        cfg = astra.astra_dict('FP')
        cfg['ProjectionDataId'] = sinogram_id
        cfg['VolumeDataId'] = volume_id
        cfg['ProjectorId'] = self.proj_id
        
        # 创建并运行投影算法
        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id)
        
        # 获取sinogram数据
        sinogram = astra.data2d.get(sinogram_id)
        
        # 清理ASTRA对象
        astra.algorithm.delete(alg_id)
        astra.data2d.delete(volume_id)
        astra.data2d.delete(sinogram_id)
        
        return sinogram
    
    def reconstruct_sirt(self, sinogram, num_iterations=10, mask=None):
        """
        使用ASTRA进行SIRT重建
        
        参数:
            sinogram: 输入sinogram
            num_iterations: 迭代次数
            mask: 只更新mask指定的区域
            
        返回:
            重建图像
        """
        return reconstruct_sirt_astra(sinogram, self.proj_geom, self.vol_geom, 
                                    num_iterations, mask)
    
    def segment_image(self, image):
        """使用当前阈值和灰度级分割图像"""
        segmented = np.zeros_like(image)
        
        # 低于第一个阈值的区域
        mask = image < self.thresholds[0]
        segmented[mask] = self.grey_levels[0]
        
        # 中间区域
        for i in range(1, len(self.thresholds)):
            mask = (image >= self.thresholds[i-1]) & (image < self.thresholds[i])
            segmented[mask] = self.grey_levels[i]
        
        # 高于最后一个阈值的区域
        mask = image >= self.thresholds[-1]
        segmented[mask] = self.grey_levels[-1]
        
        return segmented
    
    def optimize_grey_levels(self, image, thresholds):
        """
        优化灰度级(内层优化)
        
        参数:
            image: 当前重建图像
            thresholds: 当前阈值
            
        返回:
            优化后的灰度级
        """
        # 创建分割掩码
        masks = []
        masks.append(image < thresholds[0])
        for i in range(1, len(thresholds)):
            masks.append((image >= thresholds[i-1]) & (image < thresholds[i]))
        masks.append(image >= thresholds[-1]))
        
        # 计算每个分区的投影贡献
        A = np.zeros((self.sinogram.size, self.num_grey_levels))
        for i, mask in enumerate(masks):
            seg = np.zeros_like(image)
            seg[mask] = 1
            A[:, i] = self.forward_project(seg).flatten()
        
        # 解线性方程组(公式19)
        Q = A.T @ A
        c = -2 * A.T @ self.sinogram.flatten()
        
        try:
            grey_levels = np.linalg.solve(2 * Q, -c)
            # 确保灰度级有序
            grey_levels = np.sort(grey_levels)
            return grey_levels
        except np.linalg.LinAlgError:
            return self.grey_levels
    
    def optimize_thresholds(self, image, grey_levels):
        """
        优化阈值(外层优化)
        
        参数:
            image: 当前重建图像
            grey_levels: 当前灰度级
            
        返回:
            优化后的阈值
        """
        def projection_distance(t):
            # 分割图像
            segmented = self.segment_image_with_given_params(image, t, grey_levels)
            # 计算投影距离
            sino = self.forward_project(segmented)
            return np.linalg.norm(sino - self.sinogram)
        
        # 使用Nelder-Mead方法优化
        res = minimize(projection_distance, self.thresholds, method='Nelder-Mead')
        return res.x
    
    def segment_image_with_given_params(self, image, thresholds, grey_levels):
        """使用给定参数分割图像"""
        segmented = np.zeros_like(image)
        
        mask = image < thresholds[0]
        segmented[mask] = grey_levels[0]
        
        for i in range(1, len(thresholds)):
            mask = (image >= thresholds[i-1]) & (image < thresholds[i])
            segmented[mask] = grey_levels[i]
        
        mask = image >= thresholds[-1]
        segmented[mask] = grey_levels[-1]
        
        return segmented
    
    def get_boundary_pixels(self, segmented):
        """获取边界像素(与邻居不同的像素)"""
        boundary = np.zeros_like(segmented, dtype=bool)
        
        # 检查4邻域
        for i in range(1, segmented.shape[0]-1):
            for j in range(1, segmented.shape[1]-1):
                center = segmented[i, j]
                if (center != segmented[i-1, j] or center != segmented[i+1, j] or
                    center != segmented[i, j-1] or center != segmented[i, j+1]):
                    boundary[i, j] = True
        
        return boundary
    
    def reconstruct(self, num_iterations=20, sirt_iterations=10, update_params_every=5):
        """
        PDM-DART主重建算法(使用ASTRA)
        
        参数:
            num_iterations: DART迭代次数
            sirt_iterations: 每次DART迭代中的SIRT迭代次数
            update_params_every: 每隔多少次迭代更新参数
            
        返回:
            最终重建图像
        """
        # 初始SIRT重建
        self.reconstruction = self.reconstruct_sirt(self.sinogram, sirt_iterations)
        
        for k in range(num_iterations):
            print(f"Iteration {k+1}/{num_iterations}")
            
            # 每隔update_params_every次迭代更新参数
            if k % update_params_every == 0:
                # 优化灰度级
                self.grey_levels = self.optimize_grey_levels(self.reconstruction, self.thresholds)
                
                # 优化阈值
                self.thresholds = self.optimize_thresholds(self.reconstruction, self.grey_levels)
                print(f"Updated params - grey levels: {self.grey_levels}, thresholds: {self.thresholds}")
            
            # 分割当前重建
            segmented = self.segment_image(self.reconstruction)
            
            # 确定更新像素集(边界像素+随机像素)
            boundary = self.get_boundary_pixels(segmented)
            random_pixels = np.random.random(self.phantom_shape) < 0.05  # 5%随机像素
            update_mask = boundary | random_pixels
            
            # 计算残差sinogram
            fixed_pixels = np.where(~update_mask, segmented, 0)
            residual_sino = self.sinogram - self.forward_project(fixed_pixels)
            
            # 重建残差(仅更新指定像素)
            update_recon = self.reconstruct_sirt(residual_sino, sirt_iterations, update_mask)
            
            # 更新重建图像
            self.reconstruction = np.where(update_mask, update_recon, segmented)
            
            # 应用高斯平滑
            self.reconstruction = gaussian_filter(self.reconstruction, sigma=0.5)
        
        # 清理ASTRA投影器
        astra.projector.delete(self.proj_id)
        
        return self.reconstruction