

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Tuple

from enc.component.core.quantizer import (
    POSSIBLE_QUANTIZATION_NOISE_TYPE,
    POSSIBLE_QUANTIZER_TYPE,
)


MODULE_TO_OPTIMIZE = Literal["all", "arm", "upsampling", "synthesis", "latent"]


@dataclass
class TrainerPhase:
    lr: float = 1e-2
    max_itr: int = 5000
    freq_valid: int = 100
    patience: int = 10000
    quantize_model: bool = False
    schedule_lr: bool = False
    softround_temperature: Tuple[float, float] = (0.3, 0.3)
    noise_parameter: Tuple[float, float] = (1.0, 1.0)
    quantizer_noise_type: POSSIBLE_QUANTIZATION_NOISE_TYPE = "kumaraswamy"
    quantizer_type: POSSIBLE_QUANTIZER_TYPE = "softround"
    optimized_module: List[MODULE_TO_OPTIMIZE] = field(default_factory=lambda: ["all"])

    def __post_init__(self):
        if "all" in self.optimized_module:
            self.optimized_module == ["all"]

    def pretty_string(self) -> str:

        s = f'{f"{self.lr:1.2e}":^{14}}|'
        s += f"{self.max_itr:^{9}}|"
        s += f"{self.patience:^{16}}|"
        s += f"{self.freq_valid:^{13}}|"
        s += f"{self.quantize_model:^{13}}|"
        s += f"{self.schedule_lr:^{13}}|"

        softround_str= ', '.join([f'{x:1.1e}' for x in self.softround_temperature])
        s += f'{f"{softround_str}":^{18}}|'

        noise_str = ', '.join([f'{x:1.2f}' for x in self.noise_parameter])
        s += f'{f"{noise_str}":^{14}}|'
        return s

    @classmethod
    def _pretty_string_column_name(cls) -> str:
        s = f'{"Learn rate":^{14}}|'
        s += f'{"Max itr":^{9}}|'
        s += f'{"Patience [itr]":^{16}}|'
        s += f'{"Valid [itr]":^{13}}|'
        s += f'{"Quantize NN":^{13}}|'
        s += f'{"Schedule lr":^{13}}|'
        s += f'{"Softround Temp":^{18}}|'
        s += f'{"Noise":^{14}}|'
        return s

    @classmethod
    def _vertical_line_array(cls) -> str:
        s = '-' * 14 + '+'
        s += '-' * 9 + '+'
        s += '-' * 16 + '+'
        s += '-' * 13 + '+'
        s += '-' * 13 + '+'
        s += '-' * 13 + '+'
        s += '-' * 18 + '+'
        s += '-' * 14 + '+'
        return s


@dataclass
class WarmupPhase:

    candidates: int
    training_phase: TrainerPhase

    def pretty_string(self) -> str:
        s = f"|{self.candidates:^{14}}|"
        s += f"{self.training_phase.pretty_string()}"
        return s

    @classmethod
    def _pretty_string_column_name(cls) -> str:
        s = f'|{"Candidates":^{14}}|'
        s += f'{TrainerPhase._pretty_string_column_name()}'
        return s


@dataclass
class Warmup:
    phases: List[WarmupPhase] = field(default_factory=lambda: [])

    def _get_total_warmup_iterations(self) -> int:
        return sum(
            [phase.candidates * phase.training_phase.max_itr for phase in self.phases]
        )


@dataclass
class Preset:
    preset_name: str
    all_phases: List[TrainerPhase] = field(default_factory=lambda: [])
    warmup: Warmup = field(default_factory=lambda: Warmup())

    def __post_init__(self):
        flag_quantize_model = False
        for training_phase in self.all_phases:
            if training_phase.quantize_model:
                flag_quantize_model = True

        assert flag_quantize_model or len(self.all_phases) == 0, (
            f"The selected preset ({self.preset_name}) does not include "
            f" a training phase with neural network quantization.\n"
            f"{self.pretty_string()}"
        )

    def _get_total_training_iterations(self) -> int:
        return sum(
            [phase.max_itr for phase in self.all_phases]
        )

    def pretty_string(self) -> str:
        s = f"Preset: {self.preset_name:<10}\n"
        s += "-------\n"
        s += "\nWarm-up\n"
        s += "-------\n"
        s += "+" + "-" * 14 + "+" + TrainerPhase._vertical_line_array() + "\n"
        s += WarmupPhase._pretty_string_column_name() + "\n"
        s += "+" + "-" * 14 + "+" + TrainerPhase._vertical_line_array() + "\n"
        for warmup_phase in self.warmup.phases:
            s += warmup_phase.pretty_string() + "\n"
        s += "+" + "-" * 14 + "+" + TrainerPhase._vertical_line_array() + "\n"

        s += "\nMain training\n"
        s += "-------------\n"
        s += "+" + "-" * 14 + "+" + TrainerPhase._vertical_line_array() + "\n"
        s += f'|{"Phase index":^14}|{TrainerPhase._pretty_string_column_name()}\n'
        s += "+" + "-" * 14 + "+" + TrainerPhase._vertical_line_array() + "\n"
        for idx, training_phase in enumerate(self.all_phases):
            s += f"|{idx:^14}|{training_phase.pretty_string()}\n"
        s += "+" + "-" * 14 + "+" + TrainerPhase._vertical_line_array() + "\n"

        s += "\nMaximum number of iterations (warm-up / training / total):"
        warmup_max_itr = self.warmup._get_total_warmup_iterations()
        training_max_itr = self._get_total_training_iterations()
        total_max_itr  =warmup_max_itr + training_max_itr
        s += f"{warmup_max_itr:^8} / {training_max_itr:^8} / {total_max_itr:^8}\n\n"
        return s


class PresetC3x(Preset):
    def __init__(self, start_lr: float = 1e-2, n_itr_per_phase: int = 100000):
        super().__init__(preset_name="c3x")
        self.all_phases: List[TrainerPhase] = [
            TrainerPhase(
                lr=start_lr,
                max_itr=n_itr_per_phase + 600,
                patience=5000,
                optimized_module=["all"],
                schedule_lr=True,
                quantizer_type="softround",
                quantizer_noise_type="gaussian",
                softround_temperature=(0.3, 0.1),
                noise_parameter=(0.25, 0.1),
            ),
            TrainerPhase(
                lr=1.0e-4,
                max_itr=1500,
                patience=1500,
                optimized_module=["all"],
                schedule_lr=True,
                quantizer_type="ste",
                quantizer_noise_type="none",
                softround_temperature=(1e-4, 1e-4),
                noise_parameter=(1.0, 1.0),
                quantize_model=True,
            ),
            TrainerPhase(
                lr=1.0e-4,
                max_itr=1000,
                patience=50,
                quantizer_type="ste",
                quantizer_noise_type="none",
                optimized_module=["latent"],
                freq_valid=10,
                softround_temperature=(1e-4, 1e-4),
                noise_parameter=(1.0, 1.0),
            ),
        ]

        self.warmup = Warmup(
            [
                WarmupPhase(
                    candidates=5,
                    training_phase=TrainerPhase(
                        lr=start_lr,
                        max_itr=400,
                        freq_valid=400,
                        patience=100000,
                        quantize_model=False,
                        schedule_lr=False,
                        softround_temperature=(0.3, 0.3),
                        noise_parameter=(2.0, 2.0),
                        quantizer_noise_type="kumaraswamy",
                        quantizer_type="softround",
                        optimized_module=["all"],
                    )
                ),
                WarmupPhase(
                    candidates=2,
                    training_phase=TrainerPhase(
                        lr=start_lr,
                        max_itr=400,
                        freq_valid=400,
                        patience=100000,
                        quantize_model=False,
                        schedule_lr=False,
                        softround_temperature=(0.3, 0.3),
                        noise_parameter=(2.0, 2.0),
                        quantizer_noise_type="kumaraswamy",
                        quantizer_type="softround",
                        optimized_module=["all"],
                    )
                )
            ]
        )


class PresetDebug(Preset):

    def __init__(self, start_lr: float = 1e-2, n_itr_per_phase: int = 100000):
        super().__init__(preset_name="debug")
        self.all_phases: List[TrainerPhase] = [
            TrainerPhase(
                lr=start_lr,
                max_itr=50,
                patience=100000,
                optimized_module=["all"],
                schedule_lr=True,
                quantizer_type="softround",
                quantizer_noise_type="gaussian",
                softround_temperature=(0.3, 0.1),
                noise_parameter=(0.25, 0.1),
            )
        ]

        self.all_phases.append(
            TrainerPhase(
                lr=1e-4,
                max_itr=10,
                patience=10,
                optimized_module=["all"],
                quantizer_type="ste",
                quantizer_noise_type="none",
                quantize_model=True,
                softround_temperature=(1e-4, 1e-4),
                noise_parameter=(1.0, 1.0),
            )
        )

        self.all_phases.append(
            TrainerPhase(
                lr=1e-4,
                max_itr=10,
                patience=50,
                optimized_module=["latent"],
                freq_valid=5,
                quantizer_type="ste",
                quantizer_noise_type="none",
                softround_temperature=(1e-4, 1e-4),
                noise_parameter=(1.0, 1.0),
            )
        )

        self.warmup = Warmup(
            [
                WarmupPhase(candidates=3, training_phase=TrainerPhase(max_itr=10)),
                WarmupPhase(candidates=2, training_phase=TrainerPhase(max_itr=10)),
            ]
        )


AVAILABLE_PRESETS: Dict[str, Preset] = {
    "c3x": PresetC3x,
    "debug": PresetDebug,
}
