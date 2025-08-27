from enum import Enum, IntEnum

class InferStatus(IntEnum):
    Train = 0
    Prefill = 1
    Decode = 2

class TrainerProcessType(Enum):
    TrainForward = 0
    EvaluateForward = 1
    Backward = 2


class InferMode(Enum):
    Prefill = 0
    Decode = 1

class ProcessMode(Enum):
    Batch = 0
    Single = 1

class MessageType(Enum):
    InputArrival = 0
    ControlMessage = 1

class InferInputType(Enum):
    GenerateInput = 0
    StraightForward = 1
    StartInferOnly = 2
    ConfigModify = 3

class InferOutputType(Enum):
    TokensOutput = 0
    MarkIdUpdateAdded = 1
    MarkIdupdateRemoved = 2
    FinishedSignal = 3

class InteractiveSyncType(Enum):
    OutputSync = 0      # Given Length of Tokens Output
    FinishSync = 1      # Finished Some Sequences
    InputSync = 2       # New Input Arrival
    CommandSync = 3     # New Command Arrival (Notify or Remove)
    StartSync = 4       # Start Some Sequences

class InferQueueStatus(Enum):
    Running = 0
    WaitingToStop = 1
    Stopped = 2
    WaitingToBeRemoved = 3

class InputInferStatus(Enum):
    Waiting = 0
    Ready = 1
    Running = 2
    Finished = 3 # Unused

class LoopStartType(Enum):
    AsMainThread = 0
    AsChildThread = 1
    # AsAsyncThread = 2
