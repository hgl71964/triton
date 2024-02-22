import os
import subprocess
import pickle
import tempfile
import time

import random
import numpy as np

# asm (add CuAsm to PYTHONPATH!)
from CuAsm.CubinFile import CubinFile
from CuAsm.CuInsAssemblerRepos import CuInsAssemblerRepos
from CuAsm.CuInsFeeder import CuInsFeeder

from fgk.jit import search, jit
from fgk.autotuner import autotune as fgk_autotune
from fgk.utils.gpu_utils import get_gpu_name, get_gpu_cc

from absl import app
from absl import flags

# YAPF: disable
FLAGS = flags.FLAGS

# kernel
flags.DEFINE_string("f", None, "")
flags.DEFINE_string("d", None, "")
flags.DEFINE_string("save", None, "")
flags.DEFINE_integer("arch", None, "")


def dump_one(inf, outf, folder, arch):
    outf = os.path.join(folder, outf)
    command = f"cuobjdump -sass -arch {arch} {inf} > {outf}"
    
    # Run the command using subprocess
    result = subprocess.run(command, capture_output=True, text=True, cwd=folder, shell=True)
    
    # Check if the command was executed successfully
    if result.returncode == 0:
        print(f"Output for {inf}:\n{result.stdout}")
    else:
        raise Exception(f"Error running cuobjdump on {inf}:\n{result.stderr}")
    
    feeder = CuInsFeeder(outf)   
    return feeder


def main(_):
    arch = f'sm_{FLAGS.arch}'
    repos = CuInsAssemblerRepos(arch=arch)      
    # tmp_dir='/home/gh512/workspace/triton/tmp'

    # with tempfile.TemporaryDirectory(delete=True) as tmp_dir:
    with tempfile.TemporaryDirectory() as tmp_dir:

        if FLAGS.f is not None:
            inf = FLAGS.f
            outf = 'tmp_0.sass'
            feeder = dump_one(inf, outf, tmp_dir, arch)

            repos.update(feeder)                       

        elif FLAGS.d is not None:
            directory = FLAGS.d
            so_files = [f for f in os.listdir(directory) if f.endswith('.so')]

            for i, so_file in enumerate(so_files):
                inf = os.path.join(directory, so_file)
                outf = f'tmp_{i}.sass'
                feeder = dump_one(inf, outf, tmp_dir, arch)
                repos.update(feeder)                       
                break

    if FLAGS.save is not None:
        repos.save2file(os.path.join(FLAGS.save, 'Repos.'+arch+'.txt'))     


if __name__ == "__main__":
    app.run(main)
