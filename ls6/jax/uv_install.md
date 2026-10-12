# uv installation

uv should help streamline installations!

## configuring and installing uv
### Ensure uv places downloaded Python binaries and caches in $WORK
export UV_CACHE_DIR="$WORK/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$WORK/.local/share/uv/python"

### Tell the installer to place the 'uv' executable in $WORK as well
export CARGO_HOME="$WORK/.cargo"

### download and install uv
curl -LsSf https://astral.sh | sh
source ~/.bashrc

## set up JAX, lineax, and (maybe) equinox
### Move to your work directory
cd $WORK

### Create a Python 3.12 environment in $WORK
uv venv jax_env --python 3.12
source jax_env/bin/activate

### Install JAX with CUDA 12 wheels, Lineax, and Equinox
uv pip install -U "jax[cuda12]" lineax equinox

## multi-node jax and lineax configuration
I'll need to check this out.

```python
import jax
import jax.numpy as jnp
import lineax as lx
import equinox as eqx

# 1. Automatically detect cluster topology from Slurm environment variables
jax.distributed.initialize()

rank = jax.process_index()
num_devices = jax.device_count()
local_devices = jax.local_device_count()

if rank == 0:
    print(f"Total global GPUs in cluster: {num_devices}")
    print(f"Local GPUs per node: {local_devices}")

# 2. Define a multi-node Mesh for sharding data across devices
# Lonestar6 has 3 A100s per GPU node (or 2 H100s)
devices = jax.devices()
mesh = jax.sharding.Mesh(devices, ('x',))
sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('x'))

# 3. Create your linear system Ax = b
# In a real run, you would shard the generation or load chunks of data per node
size = 8192
key = jax.random.PRNGKey(0)

# Generate a random positive-definite matrix and a right-hand side vector
A_raw = jax.random.normal(key, (size, size))
A_mat = jnp.dot(A_raw, A_raw.T) + jnp.eye(size) * 10  # Ensure invertibility
b_vec = jax.random.normal(key, (size,))

# Shard the matrix and vector across all available cluster nodes
A_sharded = jax.device_put(A_mat, sharding)
b_sharded = jax.device_put(b_vec, sharding)

# 4. Use Lineax to solve Ax = b across the cluster
# Lineax operator wraps the matrix; lx.CG() is an iterative conjugate gradient solver
operator = lx.MatrixLinearOperator(A_sharded)
solution = lx.linear_solve(operator, b_sharded, solver=lx.CG(), options={"max_steps": 1000})

### Print result check on the root node
if rank == 0:
    print("Linear system successfully solved across multiple nodes!")
    print(f"Solution shape: {solution.value.shape}")
```

### slurm batch script
```bash
#!/bin/bash
#SBATCH -J jax_multinode
#SBATCH -o jax_multinode.%j.out
#SBATCH -e jax_multinode.%j.err
#SBATCH -p gpu-a100
#SBATCH -N 2                       # Number of nodes
#SBATCH -n 6                       # Total tasks (Must equal Total GPUs: 2 nodes x 3 GPUs)
#SBATCH --ntasks-per-node=3        # Exactly 3 tasks per node to match the 3 A100s
#SBATCH -t 00:20:00                # Runtime limit (HH:MM:SS)
#SBATCH -A YOUR_ALLOCATION_NAME    # Your TACC project allocation

# Force JAX to fail fast if it can't find a path to the GPUs
export JAX_PLATFORM_NAME=gpu

# Activate your $WORK-based environment
source $WORK/jax_env/bin/activate

# Execute using srun to broadcast the Python task across the Slurm allocation
srun python multinode_solve.py
```

### submit to queue
```bash
sbatch run_jax.sbatch
```

## Parallelizing lineax
This seems like a work in progress... need to push the boundary on this!

`sharp_map` seems to be preferred over `jax.pmap` now; I'll need to do some testing and reading regarding this.