import itertools
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from phantoms import create_phantom
from utils_astra import generate_sinogram_astra, create_astra_geometry, display_sinogram, add_noise, compute_rmse, compute_rnmp, save_image, reconstruct_fbp_astra, reconstruct_sirt_astra
from pdmdart_astra import PDMDARTAstra

def run_ablation():
    param_grid = {
        'num_pdm_iterations': [30],
        'sirt_iterations': [40],
        'num_grey_levels': [ 5, 10, 15],
        'update_every': [5],
        #'phantom_type' : ['basic'],
        'phantom_type': ['basic', 'resolution', 'ct','filled'],
        'noise_type': [None],
        'num_angles': [100],
        'detector_size_factor': [4]
    }

    # Generate all combinations of hyperparameters
    keys = list(param_grid.keys())
    combinations = list(itertools.product(*param_grid.values()))

    records = []

    for combo in combinations:
        settings = dict(zip(keys, combo))

        print(f"Running: {settings}")
        try:
            # Generate phantom and sinogram
            phantom = create_phantom(
                phantom_type=settings['phantom_type'],
               
                noise_type=settings['noise_type'],
                seed=42
            )
            proj_geom, vol_geom = create_astra_geometry(
                phantom.shape, num_angles=settings['num_angles'], detector_factor = settings['detector_size_factor']
            )
            sino = generate_sinogram_astra(phantom, proj_geom, vol_geom)

            # Run PDM-DART reconstruction
            reconstructor = PDMDARTAstra(
                sino, phantom, phantom.shape,
                num_angles=settings['num_angles'],
                num_grey_levels=settings['num_grey_levels'],
                detector_factor = settings['detector_size_factor']
            )
            recon = reconstructor.reconstruct(
                num_iterations=settings['num_pdm_iterations'],
                sirt_iterations=settings['sirt_iterations'],
                update_params_every=settings['update_every']
            )

            # Compute RNMP
            rnmp = compute_rnmp(phantom, recon)


            save_image(recon, f"results/pdm_grey_levels_{settings['num_grey_levels']}_{settings['phantom_type']}.png")
            # Save full config + rnmp
            record = settings.copy()
            record['rnmp'] = rnmp
            records.append(record)

            print(f"✅ Done: RNMP={rnmp:.4f}")

        except Exception as e:
            print(f"❌ Failed for {settings}: {e}")

    # Save to CSV
    df = pd.DataFrame(records)
    df.to_csv('ablation_results_fullgrid.csv', index=False)
    
    # # Optional: plot RNMP vs one hyperparam at a time
    # sns.set(style="whitegrid")
    # for param in keys:
    #     plt.figure(figsize=(6, 4))
    #     sns.lineplot(data=df, x=param, y='rnmp', marker='o')
    #     plt.title(f'Ablation on {param}')
    #     plt.xlabel(param)
    #     plt.ylabel('RNMP')
    #     plt.tight_layout()
    #     plt.savefig(f'ablation_{param}.png', dpi=150)
    #     plt.show()

if __name__ == '__main__':
    run_ablation()
