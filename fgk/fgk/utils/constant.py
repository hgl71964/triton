# YAPF: disable

# example sass opcode
# see : https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html

def get_mutatable_ops(cc):
    if cc == (8, 0):
        memory_ops = ['LDG', 'STG', 'LDS', 'LDSM']
        ban_ops = [
            'LDGDEPBAR'
            'DEPBAR',
        ]
        return memory_ops, ban_ops
    elif cc == (7, 5):
        memory_ops = [
            # load insts
            'LDG', 'LDS', 'LDSM', 'LDL', 'LD',

            # store insts
            'STG', 'STS', 'STL', 'ST',
        ]
        ban_ops = [
            'ERRBAR',
            'MEMBAR'
            'BAR',
            'DEPBAR',

            'ULDC',
        ]
        return memory_ops, ban_ops
    else:
        raise RuntimeError(f'unsupported compute capability: {cc}')
