#!/usr/bin/env -S uv run --script

from pathlib import Path
from cellpose import models
from cellpose.io import imread
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import glasbey
from tifffile import tifffile
import tqdm
from skimage.morphology import disk, binary_dilation
from skimage.segmentation import find_boundaries
import pandas as pd

def main() -> None:
    ##################
    ####### Parameters
    thickness = 5  # thickness for membrane mask dilation
    mode = "thick" # boundary mode: "thick", "inner"
    ##################

    model = models.CellposeModel(gpu=True)

    results = {
        "file_name": [],
        "num_cells": [],
        "norm_intensity": [],
    }

    # list all tif files in data
    data_path = Path('data')
    files = sorted(list(data_path.glob('*.tif')))
    print(f'Found {len(files)} files')

    for k in tqdm.tqdm(range(0, len(files))):
        imgs = [imread(files[k])]

        masks, *_ = model.eval(imgs, flow_threshold=0.4, cellprob_threshold=0.0)
        print(f"masks[0].shape: {masks[0].shape}")

        mask = masks[0]
        img = imgs[0][2]

        unique_vals = len(np.unique(mask)) - 1  # exclude background
        
        output_path = Path('output')
        output_path.mkdir(exist_ok=True, parents=True)

        p = Path(files[k])
        name = p.stem

        # save mask as tiff
        tifffile.imwrite(output_path / f'{name}_mask.tiff', mask.astype(np.uint16))

        # colormap
        num_instances = unique_vals + 1  # include background
        glasbey_palette = glasbey.create_palette(palette_size=num_instances)
        
        # Create colored mask using Glasbey palette
        color_mask = np.zeros((*mask.shape, 3))
        for j in range(1, num_instances):
            # Convert hex color to RGB values (0-1 range)
            hex_color = glasbey_palette[j]
            rgb_color = mcolors.hex2color(hex_color)
            color_mask[mask == j] = rgb_color

        # plot mask overlay raw image with no subplot
        plt.imshow(img, cmap='gray')
        plt.imshow(color_mask, alpha=0.5)
        plt.title(f'{name}: {unique_vals} cells')
        plt.axis('off')
        plt.tight_layout()

        plt.savefig(output_path / f'{name}_cell_segmentation.png')
        plt.close()

        # Erode and make membrane mask
        membrane_mask = binary_dilation(
            find_boundaries(mask, mode=mode),
            disk(thickness)
        )
        tifffile.imwrite(output_path / f'{name}_membrane_mask.tiff', membrane_mask)

        norm_intensity = np.sum(img[membrane_mask]) / np.sum(membrane_mask)

        # superimpose mambrane mask on raw image
        plt.imshow(img, cmap='gray')
        plt.imshow(membrane_mask, alpha=0.5, cmap='Reds')
        plt.title(f'{name} - intensity {norm_intensity}')
        plt.axis('off')
        plt.tight_layout()

        plt.savefig(output_path / f'{name}_membrane_mask.png')
        plt.close()

        # record results
        results["file_name"].append(files[k].name)
        results["num_cells"].append(unique_vals)
        results["norm_intensity"].append(norm_intensity)

    # save to csv
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path / 'summary_results.csv', index=False)

if __name__ == '__main__':
    main()
