from enum import Enum, IntEnum

class InferStatus(IntEnum):
    Train = 0
    Prefill = 1
    Decode = 2

class TrainerProcessType(Enum):
    TrainForward = 0
    EvaluateForward = 1
    Backward = 2

