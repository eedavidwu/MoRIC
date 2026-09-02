
from dataclasses import dataclass, field, fields
from enc.utils.presets import AVAILABLE_PRESETS, Preset


@dataclass
class FrameEncoderManager():
    preset_name: str
    start_lr: float = 1e-2
    lmbda: float = 1e-3
    n_itr: int = int(1e5)
    n_loops: int = 1

    preset: Preset = field(init=False)

    idx_best_loop: int = field(default=0, init=False)
    best_loss: float = field(default=1e6, init=False)
    loop_counter: int = field(default=0, init=False)

    total_training_time_sec: float = field(default=0., init=False)
    iterations_counter: int = field(default=0, init=False)

    def __post_init__(self):
        assert self.preset_name in AVAILABLE_PRESETS, f'Preset named {self.preset_name} does not exist.' \
            f' List of available preset:\n{list(AVAILABLE_PRESETS.keys())}.'

        self.preset = AVAILABLE_PRESETS.get(self.preset_name)(start_lr= self.start_lr, n_itr_per_phase=self.n_itr)

        flag_quantize_model = False
        for training_phase in self.preset.all_phases:
            if training_phase.quantize_model:
                flag_quantize_model = True
        assert flag_quantize_model, f'The selected preset ({self.preset_name}) does not include ' \
            f' a training phase with neural network quantization.\n{self.preset.pretty_string()}'

    def record_beaten(self, candidate_loss: float) -> bool:
        return candidate_loss < self.best_loss

    def set_best_loss(self, new_best_loss: float):
        self.best_loss = new_best_loss
        self.idx_best_loop = self.loop_counter


    def pretty_string(self) -> str:
        ATTRIBUTE_WIDTH = 25
        VALUE_WIDTH = 80

        s = 'FrameEncoderManager value:\n'
        s += '--------------------------\n'
        for k in fields(self):
            if k.name == 'preset':
                continue
            s += f'{k.name:<{ATTRIBUTE_WIDTH}}: {str(getattr(self, k.name)):<{VALUE_WIDTH}}\n'
        s += '\n'
        return s


