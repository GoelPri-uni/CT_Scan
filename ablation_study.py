import itertools
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from phantoms import generate_phantom
from utils_astra import create_astra_geometry, generate_sinogram_astra, compute_rnmp
from pdmdart_astra import PDMDARTAstra

def run_ablation():
    # default settings from your initial code
    default = {
        'num_iterations': 20,
        'sirt_iterations': 50,
        'update_every': 5,
        'phantom_type': 'resolution',
        'resolution': 512,
        'noise_type': None,
        'num_angles': 5
    }

    # values to sweep for each hyperparameter
    param_grid = {
        'num_iterations': [10, 20, 50],
        'sirt_iterations': [20, 50, 100],
        'update_every': [1, 5, 10],
        'phantom_type': ['resolution', 'ct', 'filled'],
        'resolution': [128, 256, 512],
        'noise_type': [None, 'poisson', 'gaussian'],
        'num_angles': [5, 30, 60],
    }

    records = []
    # vary one hyper‐parameter at a time
    for param, values in param_grid.items():
        for val in values:
            settings = default.copy()
            settings[param] = val

            # generate phantom & sinogram
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

            # run PDM‐DART
            reconstructor = PDMDARTAstra(sino, phantom, phantom.shape, num_grey_levels=10)
            recon = reconstructor.reconstruct(
                num_iterations=settings['num_iterations'],
                sirt_iterations=settings['sirt_iterations'],
                update_params_every=settings['update_every']
            )

            # compute RNMP
            rnmp = compute_rnmp(phantom, recon)
            records.append({
                'hyperparam': param,
                'value': val,
                'rnmp': rnmp
            })
            print(f"Done {param}={val} → rnmp={rnmp:.4f}")

    # save results
    df = pd.DataFrame(records)
    df.to_csv('ablation_results.csv', index=False)

    # plot one line per hyperparameter
    sns.set(style="whitegrid")
    for param in param_grid:
        subset = df[df['hyperparam'] == param]
        plt.figure(figsize=(6,4))
        sns.lineplot(data=subset, x='value', y='rnmp', marker='o')
        plt.title(f'Ablation study on {param}')
        plt.xlabel(param)
        plt.ylabel('RNMP')
        plt.tight_layout()
        plt.savefig(f'ablation_{param}.png', dpi=150)
        plt.show()

if __name__ == '__main__':
    run_ablation()
