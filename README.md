# Cell boundary intensity analysis

This script runs on a set of `.tif` files saved a `data` folder. The images must have
channels.




## How to install (first time)

1. First we need to install a tool called [uv](https://docs.astral.sh/uv/getting-started/installation/) to manage the python environment in the command line:
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```
2. At the end of the script `uv` suggest to run a command to make sure it is recognized in the terminal.
    ``` bash
    source $HOME/.local/bin/env
    ```
3. Navigate (`cd folder/path`) to the folder of your choice or open a terminal directly in the folder. 
4. You can download the repository directly from the website or use:
    ```bash
    git clone https://github.com/nobias-fht/cell_boundaries.git
    ```
5. Navigate to the folder:
    ```bash
    cd cell_boundaries
    ```

## How to run the script

```bash
uv run python process_data.py
```
