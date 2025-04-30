import numpy as np
import matplotlib.pyplot as plt
import astra
from skimage.transform import radon, iradon, rescale, iradon_sart
from phantoms import generate_phantom
from scipy.ndimage import gaussian_filter


def add_noise(img, noise_type="none"):
    if noise_type == "gaussian" or noise_type == "both":
        img = gaussian_filter(img, sigma=1) + np.random.normal(0, 0.03, img.shape)
        img = np.clip(img, 0, 1)
    if noise_type == "poisson" or noise_type == "both":
        img = np.random.poisson(img * 255) / 255.0
        img = np.clip(img, 0, 1)
    return img


def compute_rNMP(recon, ground_truth):
    recon_bin = (recon > 0.5).astype(int)
    gt_bin = (ground_truth > 0.5).astype(int)
    num_error = np.sum(recon_bin != gt_bin)
    non_zero_pixels = np.count_nonzero(gt_bin)
    return num_error / non_zero_pixels if non_zero_pixels > 0 else 0.0


def sirt(sinogram, theta, iterations=50, relaxation=1.0, img_shape=(512, 512)):
    """Basic SIRT implementation using numpy and radon transforms."""
    recon = np.zeros(img_shape, dtype=np.float32)

    for i in range(iterations):
        # Forward projection of current estimate
        projection = radon(recon, theta=theta, circle=True)

        # Compute difference with measured sinogram
        diff = sinogram - projection

        # Backproject the difference
        correction = iradon_sart(diff, theta=theta, image=recon, clip=False)

        # Update with relaxation
        recon += relaxation * correction

    return np.clip(recon, 0, 1)


def main():
    # ----- Config -----
    phantom_type = "resolution"    # or "ct", "filled"
    resolution = 512
    noise_type = "none"            # "none", "gaussian", "poisson", "both"
    num_angles = 5
    recon_method = "sirt"          # "fbp" or "sirt"
    sirt_iterations = 50

    # ----- Step 1: Generate Phantom -----
    # Generate the phantom image
    phantom = generate_phantom(
        phantom_type=phantom_type,
        resolution=resolution,
        noise_type=None,
        seed=42
    )

    phantom_noisy = add_noise(phantom.copy(), noise_type=noise_type)

    # ----- Step 2: Generate Sinogram -----
    # Generate the sinogram from the phantom
    theta = np.linspace(0., 180., num_angles, endpoint=False)
    sinogram = radon(phantom_noisy, theta=theta, circle=True)

    # ----- Step 3: Reconstruction -----
    # Perform reconstruction using the specified method
    if recon_method == "fbp":
        reconstruction = iradon(sinogram, theta=theta, circle=True)
    elif recon_method == "sirt":
        reconstruction = sirt(sinogram, theta, iterations=sirt_iterations, img_shape=phantom.shape)
    else:
        raise ValueError("Unsupported reconstruction method. Use 'fbp' or 'sirt'.")

    # ----- Step 4: Evaluation -----
    # Evaluate the reconstruction quality
    rmse = np.sqrt(np.mean((phantom - reconstruction) ** 2))
    r_nmp = compute_rNMP(reconstruction, phantom)
    print(f"[INFO] Method: {recon_method.upper()}, RMSE = {rmse:.4f}, rNMP = {r_nmp:.4f}")

    # ----- Step 5: Visualization -----
    # Visualize the phantom, sinogram, and reconstruction
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(phantom, cmap='gray')
    axs[0].set_title("Original Phantom")
    axs[1].imshow(sinogram, cmap='gray', aspect='auto')
    axs[1].set_title(f"Sinogram ({num_angles} angles)")
    axs[2].imshow(reconstruction, cmap='gray')
    axs[2].set_title(f"{recon_method.upper()} Reconstruction")
    for ax in axs:
        ax.axis("off")
    plt.tight_layout()
    plt.show()

    # ----- Step 6: Save -----
    # Save the phantom, sinogram, and reconstruction
    np.save("phantom.npy", phantom)
    np.save("sinogram.npy", sinogram)
    np.save(f"reconstruction_{recon_method}.npy", reconstruction)
    plt.imsave("phantom.png", phantom, cmap='gray')
    plt.imsave(f"reconstruction_{recon_method}.png", reconstruction, cmap='gray')
    plt.imsave("sinogram.png", sinogram / np.max(sinogram), cmap='gray')


if __name__ == "__main__":
    main()
