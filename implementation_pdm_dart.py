import numpy as np
import scipy.optimize
import scipy.ndimage
import matplotlib.pyplot as plt
import astra # Import ASTRA

# --- Helper Functions (segment_image, get_boundary_pixels - remain the same) ---
def segment_image(v_flat, thresholds, grey_levels):
    """Applies thresholding C_{τ,ρ}(v)."""
    segmented_v = np.zeros_like(v_flat)
    sorted_thresholds = np.sort(thresholds)
    
    if len(sorted_thresholds) == 0: # Handle single grey level case
         segmented_v[:] = grey_levels[0]
         return segmented_v

    segmented_v[v_flat < sorted_thresholds[0]] = grey_levels[0]
    for i in range(len(sorted_thresholds) - 1):
        mask = (v_flat >= sorted_thresholds[i]) & (v_flat < sorted_thresholds[i+1])
        segmented_v[mask] = grey_levels[i+1]
    segmented_v[v_flat >= sorted_thresholds[-1]] = grey_levels[-1]
    return segmented_v

def get_boundary_pixels(s_img, shape):
    """Finds boundary pixels in a segmented image (simple 4-connectivity)."""
    n_rows, n_cols = shape
    s_flat = s_img.flatten()
    boundary_mask = np.zeros_like(s_flat, dtype=bool)
    
    for r in range(n_rows):
        for c in range(n_cols):
            idx = r * n_cols + c
            val = s_flat[idx]
            is_boundary = False
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < n_rows and 0 <= nc < n_cols:
                    n_idx = nr * n_cols + nc
                    if s_flat[n_idx] != val:
                        is_boundary = True
                        break
            if is_boundary:
                boundary_mask[idx] = True
    return boundary_mask

# --- ASTRA SIRT Implementation ---

def sirt_step_astra(proj_id, v_current, p, bp_ones, relaxation=1.0):
    """Performs one iteration of the SIRT algorithm using ASTRA."""
    
    # Forward project current estimate: p_calc = W @ v_current
    sino_id, p_calc = astra.creators.create_sino(v_current, proj_id)

    # Calculate difference: diff = p - p_calc
    diff = p - p_calc
    
    # --- Row normalization approximation ---
    # This part is tricky without W. Often simplified or adjusted in practice.
    # A simple approach might ignore row normalization or use a constant factor.
    # Let's proceed without explicit row normalization for now, common in basic ASTRA SIRT.
    # More advanced: Could try to estimate row weights if needed.
    ratio = diff # Simplified: No division by W_row_sum

    # Backproject the difference: bp_ratio = W.T @ ratio
    bp_ratio_id, bp_ratio = astra.creators.create_backprojection(ratio, proj_id)

    # Column normalization and update: correction = (W.T @ ratio) / W_col_sum
    # Use bp_ones as approximation for W_col_sum
    bp_ones_safe = bp_ones.copy()
    bp_ones_safe[bp_ones_safe == 0] = 1 # Avoid division by zero
    correction = bp_ratio / bp_ones_safe
    
    v_next = v_current + relaxation * correction
    
    # Cleanup ASTRA data
    astra.data2d.delete([sino_id, bp_ratio_id])
    
    # Optional: Non-negativity
    # v_next[v_next < 0] = 0
    
    return v_next

def sirt_reconstruction_astra(vol_geom, proj_geom, p, n_iter, initial_v=None, relaxation=1.0):
    """Runs multiple SIRT iterations using ASTRA."""
    
    # Create ASTRA projector
    proj_id = astra.create_projector('cuda', proj_geom, vol_geom) # Use 'cuda' for GPU
    
    if initial_v is None:
        v_current = np.zeros((vol_geom['GridRowCount'], vol_geom['GridColCount']), dtype=np.float32)
    else:
        v_current = initial_v.astype(np.float32).copy()
        
    # Precompute backprojection of ones for normalization
    print("  Calculating SIRT normalization (bp_ones)...")
    sino_ones_id, sino_ones = astra.creators.create_sino(np.ones_like(v_current), proj_id)
    bp_ones_id, bp_ones = astra.creators.create_backprojection(np.ones(p.shape, dtype=np.float32), proj_id)
    astra.data2d.delete([sino_ones_id, bp_ones_id])
    print("  Normalization calculated.")

    print("Running SIRT...")
    for i in range(n_iter):
        v_current = sirt_step_astra(proj_id, v_current, p, bp_ones, relaxation)
        if (i+1) % 10 == 0:
            print(f"  SIRT Iteration {i+1}/{n_iter}")
            
    # Cleanup projector
    astra.projector.delete(proj_id)
            
    return v_current

# --- ASTRA PDM Implementation ---

def compute_projection_distance_astra(thresholds, v_flat, proj_id, p, num_grey_levels, vol_shape):
    """
    Objective function for PDM outer loop using ASTRA.
    Calculates ||W C_{τ,ρ_opt(τ)}(v) - p||²
    """
    n_pixels = v_flat.shape[0]
    m_projections = p.shape[0]
    v_current = v_flat.reshape(vol_shape) # Reshape for ASTRA if needed

    # --- Inner Loop: Estimate Optimal Grey Levels ρ for given τ ---
    partition_labels = np.zeros(n_pixels, dtype=int)
    sorted_thresholds = np.sort(thresholds)
    
    if len(sorted_thresholds) == 0: # Handle single grey level
        partition_labels[:] = 0
    else:
        partition_labels[v_flat < sorted_thresholds[0]] = 0
        for i in range(len(sorted_thresholds) - 1):
            mask = (v_flat >= sorted_thresholds[i]) & (v_flat < sorted_thresholds[i+1])
            partition_labels[mask] = i + 1
        partition_labels[v_flat >= sorted_thresholds[-1]] = num_grey_levels - 1

    # Calculate matrix A columns using ASTRA forward projection
    # A[:, t] = W @ s_t
    A = np.zeros((p.shape[0] * p.shape[1], num_grey_levels), dtype=np.float32) # Sinogram shape x num_levels
    astra_ids_to_delete = []
    
    for t in range(num_grey_levels):
        s_t_flat = (partition_labels == t).astype(np.float32)
        s_t_img = s_t_flat.reshape(vol_shape)
        sino_st_id, sino_st = astra.creators.create_sino(s_t_img, proj_id)
        A[:, t] = sino_st.flatten() # Store flattened sinogram as column
        astra_ids_to_delete.append(sino_st_id)
        
    astra.data2d.delete(astra_ids_to_delete) # Clean up intermediate sinograms

    # Calculate Q and c (same math as before, using the calculated A)
    p_flat = p.flatten()
    Q = A.T @ A
    c = -2 * p_flat.T @ A # c is a row vector

    # Solve 2 * Q * ρ = -c^T for ρ (Eq. 19)
    try:
        epsilon = 1e-6
        rho_opt = np.linalg.solve(2 * Q + epsilon * np.identity(num_grey_levels), -c.T).flatten()
    except np.linalg.LinAlgError:
        print("Warning: Linear system for rho_opt might be singular. Using pseudo-inverse.")
        try:
            rho_opt = (np.linalg.pinv(2 * Q + epsilon * np.identity(num_grey_levels)) @ (-c.T)).flatten()
        except np.linalg.LinAlgError:
             print("Error: Could not solve for rho_opt even with pseudo-inverse.")
             return np.inf, np.zeros(num_grey_levels)

    # Calculate Projection Distance with optimal rho
    projection_diff = A @ rho_opt - p_flat
    distance_sq = np.sum(projection_diff**2)

    return distance_sq, rho_opt

def estimate_segmentation_parameters_pdm_astra(v_img, proj_id, p, num_grey_levels, initial_thresholds):
    """ PDM outer loop using ASTRA """
    v_flat = v_img.flatten()
    vol_shape = v_img.shape

    def objective_func_astra(thresholds, v_flat_in, proj_id_in, p_in, num_levels_in, shape_in):
        dist_sq, _ = compute_projection_distance_astra(thresholds, v_flat_in, proj_id_in, p_in, num_levels_in, shape_in)
        # Add penalty if thresholds get too close or out of order? Optional.
        # if len(thresholds) > 1 and np.any(np.diff(np.sort(thresholds)) < 1e-3):
        #     return dist_sq + 1e6 # Penalize collapsed thresholds
        return dist_sq

    print("  Optimizing thresholds (PDM Outer Loop)...")
    result = scipy.optimize.minimize(
        objective_func_astra,
        initial_thresholds,
        args=(v_flat, proj_id, p, num_grey_levels, vol_shape),
        method='Nelder-Mead',
        options={'xatol': 1e-3, 'fatol': 1e-3, 'disp': False} # Adjust tolerance
    )

    if not result.success:
        print(f"  Warning: Threshold optimization did not converge: {result.message}")
        
    tau_opt = result.x
    
    # Final call to get optimal rho for the optimal tau
    final_dist_sq, rho_opt = compute_projection_distance_astra(tau_opt, v_flat, proj_id, p, num_grey_levels, vol_shape)
    
    print(f"  Optimal Thresholds (τ): {np.sort(tau_opt)}")
    print(f"  Optimal Grey Levels (ρ): {rho_opt}") 
    print(f"  Minimized Projection Distance^2: {final_dist_sq}")
    
    # Sort tau and rho together based on tau order
    if len(tau_opt) > 0:
        sorted_tau_indices = np.argsort(tau_opt)
        tau_opt_sorted = tau_opt[sorted_tau_indices]
        # Re-run inner loop solver with sorted tau to get correctly ordered rho
        _, rho_opt_sorted = compute_projection_distance_astra(tau_opt_sorted, v_flat, proj_id, p, num_grey_levels, vol_shape)
    else: # Single grey level case
        tau_opt_sorted = tau_opt
        rho_opt_sorted = rho_opt

    return tau_opt_sorted, rho_opt_sorted

# --- PDM-DART Main Algorithm with ASTRA ---

def pdm_dart_astra(
    vol_geom, proj_geom, p, # ASTRA geometries and projection data
    num_grey_levels,
    n_dart_iterations,
    n_sirt_init=50,
    n_sirt_residual=10,
    relaxation_sirt=1.0,
    boundary_fraction_r=0.01,
    smoothing_sigma=0.5,
    pdm_update_frequency=1
    ):
    """Performs PDM-DART reconstruction using ASTRA."""
    
    shape = (vol_geom['GridRowCount'], vol_geom['GridColCount'])
    n_pixels = shape[0] * shape[1]

    # Initial SIRT reconstruction
    print("Performing initial SIRT reconstruction...")
    v_k_img = sirt_reconstruction_astra(vol_geom, proj_geom, p, n_sirt_init, relaxation=relaxation_sirt)
    
    # Initial threshold/grey level guess
    min_val, max_val = np.min(v_k_img), np.max(v_k_img)
    if num_grey_levels > 1:
        padding = (max_val - min_val) * 0.05 
        tau_k = np.linspace(min_val + padding, max_val - padding, num_grey_levels - 1)
    else:
        tau_k = np.array([])
    rho_k = np.linspace(min_val, max_val, num_grey_levels)
    
    # Create ASTRA projector for the main loop
    proj_id = astra.create_projector('cuda', proj_geom, vol_geom)
    
    # Precompute bp_ones for residual SIRT steps
    print("Calculating normalization for residual SIRT...")
    bp_ones_id, bp_ones = astra.creators.create_backprojection(np.ones_like(p), proj_id)
    astra.data2d.delete(bp_ones_id) # Don't need the ID anymore
    print("Normalization calculated.")

    history = {'v': [v_k_img.copy()], 'tau': [], 'rho': []}

    print("\nStarting PDM-DART Iterations...")
    for k in range(n_dart_iterations):
        print(f"\n--- DART Iteration {k+1}/{n_dart_iterations} ---")
        v_k_flat = v_k_img.flatten()

        # --- Step 1: PDM Segmentation ---
        if (k % pdm_update_frequency == 0) or k == 0:
            tau_k, rho_k = estimate_segmentation_parameters_pdm_astra(
                v_k_img, proj_id, p, num_grey_levels, tau_k
            )
        else:
             print("  Skipping PDM update this iteration.")
             
        s_k_flat = segment_image(v_k_flat, tau_k, rho_k)
        s_k_img = s_k_flat.reshape(shape)
        
        history['tau'].append(tau_k.copy())
        history['rho'].append(rho_k.copy())

        # --- Step 2: Identify Pixels to Update (U_k) ---
        boundary_mask = get_boundary_pixels(s_k_img, shape)
        non_boundary_indices = np.where(~boundary_mask)[0]
        num_random_update = int(boundary_fraction_r * len(non_boundary_indices))
        if len(non_boundary_indices) > 0 and num_random_update > 0:
             random_indices = np.random.choice(non_boundary_indices, num_random_update, replace=False)
        else:
             random_indices = np.array([], dtype=int) # Handle cases with no non-boundary pixels

        U_k_mask = boundary_mask.copy()
        U_k_mask[random_indices] = True
        U_k_indices = np.where(U_k_mask)[0]
        F_k_indices = np.where(~U_k_mask)[0]

        # --- Step 3: Calculate Residual Sinogram (r_k) ---
        f_k_flat = np.zeros_like(v_k_flat, dtype=np.float32)
        f_k_flat[F_k_indices] = s_k_flat[F_k_indices]
        f_k_img = f_k_flat.reshape(shape)
        
        print("  Calculating residual sinogram...")
        p_fixed_id, p_fixed = astra.creators.create_sino(f_k_img, proj_id)
        r_k = p - p_fixed
        astra.data2d.delete(p_fixed_id)

        # --- Step 4: Update Reconstruction (v_{k+1}) ---
        print(f"  Updating {len(U_k_indices)} free pixels using SIRT on residual...")
        # Initial guess for residual update is 0
        v_residual_init = np.zeros(shape, dtype=np.float32) 
        
        # Run SIRT steps on the residual
        v_update_img = v_residual_init.copy()
        for _ in range(n_sirt_residual):
             v_update_img = sirt_step_astra(proj_id, v_update_img, r_k, bp_ones, relaxation_sirt)
        
        # Combine fixed part and update for free pixels
        v_next_flat = f_k_flat.copy()
        v_update_flat = v_update_img.flatten()
        v_next_flat[U_k_indices] += v_update_flat[U_k_indices]
        v_next_img = v_next_flat.reshape(shape)
        
        # Optional: Apply smoothing
        if smoothing_sigma > 0:
            print(f"  Applying Gaussian smoothing (sigma={smoothing_sigma})...")
            v_k_img = scipy.ndimage.gaussian_filter(v_next_img, sigma=smoothing_sigma)
        else:
            v_k_img = v_next_img

        history['v'].append(v_k_img.copy())

    # Cleanup main projector
    astra.projector.delete(proj_id)
    
    print("\nPDM-DART Finished.")
    return v_k_img, history


# --- Example Usage with ASTRA ---
if __name__ == "__main__":
    # 1. Define Phantom and ASTRA Geometries
    print("Setting up simulation...")
    img_size = 64 # Slightly larger size
    shape = (img_size, img_size)
    n_pixels = img_size * img_size
    
    # Create phantom (same as before)
    phantom = np.zeros(shape, dtype=np.float32)
    
    center_x, center_y = img_size // 2, img_size // 2
    radius = img_size // 3
    square_half_size = img_size // 6
    y, x = np.ogrid[:img_size, :img_size]
    grey1 = 0.0 
    grey2 = 1.0 
    mask_circle = (x - center_x)**2 + (y - center_y)**2 <= radius**2
    phantom[mask_circle] = grey2
    grey3 = 2.0 
    phantom[center_y-square_half_size:center_y+square_half_size, 
            center_x-square_half_size:center_x+square_half_size] = grey3
    num_grey_levels = 3

    # Define ASTRA geometries (Parallel Beam 2D)
    det_count = int(np.sqrt(2) * img_size)
    num_angles = 45 # Fewer angles for faster demo
    angles = np.linspace(0, np.pi, num_angles, endpoint=False) # Use radians for ASTRA

    vol_geom = astra.create_vol_geom(img_size, img_size)
    proj_geom = astra.create_proj_geom('parallel', 1.0, det_count, angles)

    # Create projector for simulation
    proj_id_sim = astra.create_projector('cuda', proj_geom, vol_geom)

    # Generate projection data using ASTRA
    sino_id, p = astra.creators.create_sino(phantom, proj_id_sim)
    print(f"Simulated sinogram shape: {p.shape}")

    # Add noise
    noise_level = 0.02
    p_noisy = p + noise_level * np.max(p) * np.random.randn(*p.shape)
    p_noisy = p_noisy.astype(np.float32) # Ensure float32 for ASTRA

    # Cleanup simulation projector and data ID
    astra.data2d.delete(sino_id)
    astra.projector.delete(proj_id_sim)

    # 2. Run PDM-DART with ASTRA
    final_recon, history = pdm_dart_astra(
        vol_geom=vol_geom, 
        proj_geom=proj_geom, 
        p=p_noisy, 
        num_grey_levels=num_grey_levels,
        n_dart_iterations=20, # More iterations might be needed
        n_sirt_init=50,
        n_sirt_residual=10,
        relaxation_sirt=0.8, # May need tuning
        boundary_fraction_r=0.05,
        smoothing_sigma=0.6, # May need tuning
        pdm_update_frequency=1 
    )

    # 3. Visualize Results (same plotting code as before)
    print("Plotting results...")
    plt.figure(figsize=(18, 10))

    plt.subplot(2, 3, 1)
    plt.imshow(phantom, cmap='gray', vmin=min(grey1, grey2, grey3), vmax=max(grey1, grey2, grey3))
    plt.title("Original Phantom")
    plt.colorbar()

    plt.subplot(2, 3, 2)
    initial_recon = history['v'][0]
    plt.imshow(initial_recon, cmap='gray')
    plt.title(f"Initial SIRT ({50} iters)")
    plt.colorbar()
    
    plt.subplot(2, 3, 3)
    plt.imshow(final_recon, cmap='gray')
    plt.title(f"Final PDM-DART ({20} iters)")
    plt.colorbar()

    plt.subplot(2, 3, 4)
    iterations = np.arange(len(history['tau']))
    taus = np.array(history['tau'])
    if taus.ndim == 2 and taus.shape[1] > 0: # Check if thresholds exist
        for i in range(taus.shape[1]):
             plt.plot(iterations, taus[:, i], 'o-', label=f'τ_{i+1}')
        plt.legend()
    plt.title("Estimated Thresholds (τ) vs Iteration")
    plt.xlabel("DART Iteration")
    plt.ylabel("Threshold Value")
    plt.grid(True)

    plt.subplot(2, 3, 5)
    rhos = np.array(history['rho'])
    if rhos.ndim == 2 and rhos.shape[1] > 0: # Check if grey levels exist
        for i in range(rhos.shape[1]):
             plt.plot(iterations, rhos[:, i], 's-', label=f'ρ_{i+1}')
        plt.legend()
    plt.title("Estimated Grey Levels (ρ) vs Iteration")
    plt.xlabel("DART Iteration")
    plt.ylabel("Grey Level Value")
    plt.grid(True)
    
    plt.subplot(2, 3, 6)
    # Show final segmented image
    final_v_flat = final_recon.flatten()
    # Handle case where optimization might fail on last iter
    if history['tau']: 
        final_tau = history['tau'][-1]
        final_rho = history['rho'][-1]
        final_segmented = segment_image(final_v_flat, final_tau, final_rho).reshape(shape)
    else: # Fallback if no PDM steps completed
        final_segmented = final_recon 
    plt.imshow(final_segmented, cmap='gray', vmin=min(grey1, grey2, grey3), vmax=max(grey1, grey2, grey3))
    plt.title("Final Segmented Image")
    plt.colorbar()

    plt.tight_layout()
    plt.show()