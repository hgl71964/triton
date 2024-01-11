from copy import deepcopy
from abc import ABC, abstractmethod

from fgk.utils.constant import get_mutatable_ops
from fgk.utils.gpu_utils import get_gpu_cc


class Sample(ABC):

    def __init__(self, kernel_section: list[str], engine):
        self.kernel_section = deepcopy(kernel_section)
        self.engine = engine

        self.candidates = []  # list of index mutable
        self.dims = None
        self._perf = None
        self.actions = []

    def __eq__(self, other):
        if not isinstance(other, Sample):
            return False
        if not len(self.kernel_section) == len(other.kernel_section):
            return False

        # an optimization for approximate equality
        for i in range(len(self.kernel_section)):
            # for i in range(1000):
            if i > len(self.kernel_section):
                break
            if not self.kernel_section[i] == other.kernel_section[i]:
                return False
        return True

    def __hash__(self):
        # approximate hash
        # concatenated_string = ''.join(self.kernel_section[:1000])
        concatenated_string = ''.join(self.kernel_section)
        return hash(concatenated_string)

    def __len__(self):
        assert self.dims is not None, f'no dims'
        return self.dims

    @property
    def perf(self):
        return self._perf

    @perf.setter
    def perf(self, value):
        self._perf = value

    def get_mutable(self) -> list[int]:
        if self.dims is not None:
            return self.candidates

        # determine which lines are possible to mutate
        # e.g. LDG, STG, and they should not cross the boundary of a label or
        # LDGDEPBAR or BAR.SYNC or rw dependencies
        lines = []
        for i, line in enumerate(self.kernel_section):
            line = line.strip()
            # skip headers
            if len(line) > 0 and line[0] == '[':
                out = self.engine.decode(line)
                ctrl_code, _, _, opcode, _, _ = out

                # opcode is like: LDG.E.128.SYS
                # i.e. {inst}.{modifier*}
                memory_ops, ban_ops = get_mutatable_ops(get_gpu_cc())
                ban = False
                for op in ban_ops:
                    if op in opcode:
                        ban = True
                        break
                if ban:
                    # print(f'ban {ctrl_code} {opcode}')
                    continue

                for op in memory_ops:
                    if op in opcode:
                        # print(f'mutable {ctrl_code} {opcode}')
                        self.candidates.append(i)
                        lines.append(line)
                        break

        # dimension of the optimization problem
        self.dims = len(self.candidates)
        return self.candidates

    @abstractmethod
    def apply(self, index, action):
        pass

    def apply_all(self, indexes, actions):
        self.actions = actions
        for index, action in zip(indexes, actions):
            self.apply(index, action)