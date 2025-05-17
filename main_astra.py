import numpy as np
import matplotlib.pyplot as plt

from phantoms import create_phantom # LABEdits
from utils_astra import generate_sinogram_astra, create_astra_geometry, display_sinogram, add_noise, compute_rmse, compute_rnmp, save_image, reconstruct_fbp_astra, reconstruct_sirt_astra
from pdmdart_astra import PDMDARTAstra

def main():
    # 1. Generate the phantom
    print("Generating phantom...")
    phantom = create_phantom("basic") #basic,  "resolution", "ct", "filled"
    
    
    phantom = phantom.astype(np.float32)
    phantom /= phantom.max()
   
    num_angles = 100  # Use 30 projection angles
    proj_geom, vol_geom = create_astra_geometry(phantom.shape, num_angles)

    
    # 3. Generate the sinogram
    print("Generating sinogram using ASTRA...")
    sinogram = generate_sinogram_astra(phantom, proj_geom, vol_geom)  # Pass vol_geom here
    #sinogram = add_noise(sinogram, noise_type='poisson', noise_level=1.0)
    

    # Display the phantom and sinogram

    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    plt.imshow(phantom, cmap='gray')
    plt.title("Original Phantom")
    plt.colorbar()
    
    plt.subplot(122)
    display_sinogram(sinogram)
    plt.tight_layout()
    plt.show()

    # Save the phantom
    save_image(phantom, "results/phantom.png")

    # Save the sinogram
    save_image(sinogram, "results/sinogram.png")

    # 4. Perform SIRT reconstruction
    print("Performing SIRT reconstruction...")

    sirt_reconstruction = reconstruct_sirt_astra(sinogram, proj_geom, vol_geom, num_iterations=40)
  
    save_image(sirt_reconstruction, "results/sirt_reconstruction.png")

    # 5. Perform PDM-DART reconstruction
    print("Starting PDM-DART reconstruction with ASTRA...")
    reconstructor = PDMDARTAstra(sinogram, phantom, phantom.shape, num_grey_levels=10) #LABEdits - sending phantom
    reconstruction = reconstructor.reconstruct(num_iterations=20, sirt_iterations=20)
    save_image(reconstruction, "results/pdm_dart_reconstruction.png")

    # 6. Evaluate reconstruction quality
    fbp_rmse = compute_rmse(phantom, sirt_reconstruction)
    fbp_rnmp = compute_rnmp(phantom, sirt_reconstruction)
    pdm_rmse = compute_rmse(phantom, reconstruction)
    pdm_rnmp = compute_rnmp(phantom, reconstruction)

    print(f"FBP RMSE: {fbp_rmse:.4f}, FBP rNMP: {fbp_rnmp:.4f}")
    print(f"PDM-DART RMSE: {pdm_rmse:.4f}, PDM-DART rNMP: {pdm_rnmp:.4f}")

    # 7. Display reconstruction results
    plt.figure(figsize=(15, 5))
    plt.subplot(141)
    plt.imshow(phantom, cmap='gray')
    plt.title("Original Phantom")
    plt.colorbar()

    plt.subplot(142)
    plt.imshow(sirt_reconstruction, cmap='gray')
    plt.title("SIRT Reconstruction")
    plt.colorbar()

    plt.subplot(143)
    plt.imshow(reconstruction, cmap='gray')
    plt.title("PDM-DART Reconstruction")
    plt.colorbar()

    plt.subplot(144)
    plt.imshow(np.abs(phantom - reconstruction), cmap='hot')
    plt.title("Absolute Difference")
    plt.colorbar()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()