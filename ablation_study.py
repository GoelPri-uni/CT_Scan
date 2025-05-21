import itertools
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from phantoms import generate_phantom
from utils_astra import create_astra_geometry, generate_sinogram_astra, compute_rnmp
from pdmdart_astra import PDMDARTAstra

def run_ablation():
    param_grid = {
        'num_iterations': [10, 20],
        'sirt_iterations': [40, 80],
        'update_every': [1, 5],
        'phantom_type': ['basic'],
        'resolution': [256, 512],
        'noise_type': ['gaussian', 'poisson'],
        'num_angles': [100, 180]
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
            phantom = generate_phantom(
                phantom_type=settings['phantom_type'],
                resolution=settings['resolution'],
                noise_type=settings['noise_type'],
                seed=42
            )
            proj_geom, vol_geom = create_astra_geometry(
                phantom.shape, num_angles=settings['num_angles']
            )
            sino = generate_sinogram_astra(phantom, proj_geom, vol_geom)

            # Run PDM-DART reconstruction
            reconstructor = PDMDARTAstra(
                sino, phantom, phantom.shape,
                num_angles=settings['num_angles'],
                num_grey_levels=10
            )
            recon = reconstructor.reconstruct(
                num_iterations=settings['num_iterations'],
                sirt_iterations=settings['sirt_iterations'],
                update_params_every=settings['update_every']
            )

            # Compute RNMP
            rnmp = compute_rnmp(phantom, recon)

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

    # Optional: plot RNMP vs one hyperparam at a time
    sns.set(style="whitegrid")
    for param in keys:
        plt.figure(figsize=(6, 4))
        sns.lineplot(data=df, x=param, y='rnmp', marker='o')
        plt.title(f'Ablation on {param}')
        plt.xlabel(param)
        plt.ylabel('RNMP')
        plt.tight_layout()
        plt.savefig(f'ablation_{param}.png', dpi=150)
        plt.show()

if __name__ == '__main__':
    run_ablation()
