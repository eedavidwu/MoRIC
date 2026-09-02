

import copy
import os
import subprocess
import time
from typing import Dict, List, Tuple

import torch
from enc.utils.manager import FrameEncoderManager
from enc.component.coolchic import CoolChicEncoderParameter
from enc.component.frame import FrameEncoder, load_frame_encoder
from enc.training.quantizemodel import quantize_model
from enc.training.test import test
from enc.training.train import train
from enc.training.warmup import warmup
from enc.utils.codingstructure import CodingStructure, Frame, FrameData
from enc.utils.misc import POSSIBLE_DEVICE, TrainingExitCode, is_job_over, mem_info
from enc.utils.yuv import load_frame_data_from_file

class VideoEncoder():

    def __init__(
        self,
        coding_structure: CodingStructure,
        shared_coolchic_parameter: CoolChicEncoderParameter,
        shared_frame_encoder_manager: FrameEncoderManager,
    ):

        self.coding_structure = coding_structure
        self.shared_coolchic_parameter = shared_coolchic_parameter
        self.shared_frame_encoder_manager = shared_frame_encoder_manager

        self.all_frame_encoders: Dict[
            str, Tuple[FrameEncoder, FrameEncoderManager]
        ] = {}


    def encode(
        self,
        path_original_sequence: str,
        device: POSSIBLE_DEVICE,
        workdir: str,
        job_duration_min: int = -1,
    ) -> TrainingExitCode:
        start_time = time.time()
        n_frames = self.coding_structure.get_number_of_frames()

        for idx_coding_order in range(n_frames):
            frame = self.coding_structure.get_frame_from_coding_order(idx_coding_order)

            if frame.already_encoded:
                continue

            frame.data = load_frame_data_from_file(
                path_original_sequence, frame.display_order
            )
            frame.refs_data = self.get_ref_data(frame)

            frame_workdir = self.get_frame_workdir(workdir, frame.display_order)

            current_coolchic_parameter = copy.deepcopy(self.shared_coolchic_parameter)
            current_coolchic_parameter.set_image_size(frame.data.img_size)
            current_coolchic_parameter.encoder_gain = (
                16 if frame.frame_type == "I" else 16
            )

            match frame.frame_type:
                case "I":
                    n_output_synthesis = 3
                case "P":
                    n_output_synthesis = 6
                case "B":
                    n_output_synthesis = 9
                case _:
                    print(
                        f"Unknown frame_type {frame.frame_type}"
                    )

            current_coolchic_parameter.layers_synthesis = [
                lay.replace("X", str(n_output_synthesis))
                for lay in current_coolchic_parameter.layers_synthesis
            ]


            if str(idx_coding_order) in self.all_frame_encoders:
                _, frame_encoder_manager = (
                    self.all_frame_encoders.get(str(idx_coding_order))
                )

            else:
                print(
                    "-" * 80 + "\n"
                    + f'{" " * 12} Coding frame {frame.coding_order + 1} / {n_frames} '
                    + f"- Display order: {frame.display_order} - "
                    + f"Coding order: {frame.coding_order}\n"
                    + "-" * 80
                )

                frame_encoder_manager = copy.deepcopy(
                    self.shared_frame_encoder_manager
                )
                frame_encoder_manager.lmbda = self.get_lmbda_from_depth(
                    frame.depth, self.shared_frame_encoder_manager.lmbda
                )

                frame_encoder_manager.frame_type = frame.frame_type


                subprocess.call(f"mkdir -p {frame_workdir}", shell=True)

                print(f"\n{frame_encoder_manager.pretty_string()}")
                print(f"{current_coolchic_parameter.pretty_string()}")
                print(f"{frame_encoder_manager.preset.pretty_string()}")


            for index_loop in range(
                frame_encoder_manager.loop_counter,
                frame_encoder_manager.n_loops,
            ):
                print(
                    "-" * 80
                    + "\n"
                    + f'{" " * 30} Training loop {frame_encoder_manager.loop_counter + 1} / '
                    + f"{frame_encoder_manager.n_loops}\n"
                    + "-" * 80
                )

                frame.to_device(device)

                n_initial_warmup_candidate = (
                    frame_encoder_manager.preset.warmup.phases[
                        0
                    ].candidates
                )
                list_candidates = [
                    FrameEncoder(
                        coolchic_encoder_param=current_coolchic_parameter,
                        frame_type=frame.frame_type,
                        frame_data_type=frame.data.frame_data_type,
                        bitdepth=frame.data.bitdepth
                    )
                    for _ in range(n_initial_warmup_candidate)
                ]

                with open(f"{frame_workdir}/archi.txt", "w") as f_out:
                    f_out.write(str(list_candidates[0].coolchic_encoder) + "\n\n")
                    f_out.write(list_candidates[0].coolchic_encoder.str_complexity() + "\n")

                
                frame_encoder = warmup(
                    frame_encoder_manager=frame_encoder_manager,
                    list_candidates=list_candidates,
                    frame=frame,
                    device=device,
                )
                
                frame_encoder.to_device(device)

                for idx_phase, training_phase in enumerate(frame_encoder_manager.preset.all_phases):
                    print(f'{"-" * 30} Training phase: {idx_phase:>2} {"-" * 30}\n')
                    mem_info("Training phase " + str(idx_phase))
                    frame_encoder = train(
                        frame_encoder=frame_encoder,
                        frame=frame,
                        frame_encoder_manager=frame_encoder_manager,
                        start_lr=training_phase.lr,
                        cosine_scheduling_lr=training_phase.schedule_lr,
                        max_iterations=training_phase.max_itr,
                        frequency_validation=training_phase.freq_valid,
                        patience=training_phase.patience,
                        optimized_module=training_phase.optimized_module,
                        quantizer_type=training_phase.quantizer_type,
                        quantizer_noise_type=training_phase.quantizer_noise_type,
                        softround_temperature=training_phase.softround_temperature,
                        noise_parameter=training_phase.noise_parameter,
                    )

                    if training_phase.quantize_model:
                        frame_encoder.coolchic_encoder._store_full_precision_param()
                        frame_encoder = quantize_model(
                            frame_encoder,
                            frame,
                            frame_encoder_manager,
                        )

                    phase_results = test(
                        frame_encoder,
                        frame,
                        frame_encoder_manager,
                    )

                    print("\nResults at the end of the phase:")
                    print("--------------------------------")
                    print(
                        f'\n{phase_results.pretty_string(show_col_name=True, mode="short")}\n'
                    )

                loop_results = test(
                    frame_encoder,
                    frame,
                    frame_encoder_manager,
                )

                path_results_log = f"{frame_workdir}results_loop_{frame_encoder_manager.loop_counter + 1}.tsv"
                with open(path_results_log, "w") as f_out:
                    f_out.write(
                        loop_results.pretty_string(show_col_name=True, mode="all") + "\n"
                    )

                if frame_encoder_manager.record_beaten(loop_results.loss):
                    print(f'Best loss beaten at loop {frame_encoder_manager.loop_counter + 1}')
                    print(f'Previous best loss: {frame_encoder_manager.best_loss * 1e3 :.6f}')
                    print(f'New best loss     : {loop_results.loss.cpu().item() * 1e3 :.6f}')

                    frame_encoder_manager.set_best_loss(loop_results.loss.cpu().item())

                    with open(f'{frame_workdir}results_best.tsv', 'w') as f_out:
                        f_out.write(loop_results.pretty_string(show_col_name=True, mode='all') + '\n')
                    self.concat_results_file(workdir)

                    best_frame_encoder = frame_encoder

                else:
                    best_frame_encoder = self.all_frame_encoders[str(frame.coding_order)][0]

                frame_encoder_manager.loop_counter += 1

                self.all_frame_encoders[str(frame.coding_order)] = (
                    copy.deepcopy(best_frame_encoder),
                    copy.deepcopy(frame_encoder_manager)
                )

                print('End of training loop\n\n')

                self.save(f'{workdir}video_encoder.pt')
                frame.data = load_frame_data_from_file(
                    path_original_sequence, frame.display_order
                )
                frame.refs_data = self.get_ref_data(frame)

                if is_job_over(start_time=start_time, max_duration_job_min=job_duration_min):
                    return TrainingExitCode.REQUEUE

            self.coding_structure.set_encoded_flag(
                coding_order=frame.coding_order, flag_value=True
            )
            print(self.coding_structure.pretty_string())
            self.save(f'{workdir}video_encoder.pt')

        return TrainingExitCode.END

    def get_frame_workdir(self, workdir: str, frame_display_order: int) -> str:
        return f"{workdir}/frame_{str(frame_display_order).zfill(3)}/"

    def concat_results_file(self, workdir: str) -> None:
        list_results_file = []
        for idx_display_order in range(self.coding_structure.get_number_of_frames()):
            cur_res_file = (
                self.get_frame_workdir(workdir, idx_display_order) + "results_best.tsv"
            )
            if not os.path.isfile(cur_res_file):
                continue

            list_results_file.append(cur_res_file)

        out_path = workdir + "results_best.tsv"

        subprocess.call(f"rm -f {out_path}", shell=True)
        for idx, frame_path in enumerate(list_results_file):
            if idx == 0:
                subprocess.call(f"cat {frame_path} >> {out_path}", shell=True)
            else:
                subprocess.call(
                    f"cat {frame_path} | head -2 | tail -1 >> {out_path}", shell=True
                )

    @torch.no_grad()
    def get_ref_data(self, frame: Frame) -> List[FrameData]:

        ref_data = []

        for idx_ref in frame.index_references:
            ref_frame = self.coding_structure.get_frame_from_display_order(idx_ref)

            if ref_frame.decoded_data is not None:
                pass
            else:
                ref_frame.refs_data = self.get_ref_data(ref_frame)
                print(
                    f"get_ref_data(): Decoding frame {ref_frame.display_order:<3}..."
                )

                frame_encoder, _ = self.all_frame_encoders.get(str(ref_frame.coding_order))

                frame_encoder.set_to_eval()
                frame_encoder.to_device("cpu")

                ref_frame.upsample_reference_to_444()

                frame_encoder_out = frame_encoder.forward(
                    reference_frames=[ref_i.data for ref_i in ref_frame.refs_data],
                    quantizer_noise_type="none",
                    quantizer_type="hardround",
                    AC_MAX_VAL=-1,
                    flag_additional_outputs=False,
                )

                ref_frame.set_decoded_data(
                    FrameData(
                        frame_encoder.bitdepth,
                        frame_encoder.frame_data_type,
                        frame_encoder_out.decoded_image,
                    )
                )

            ref_data.append(ref_frame.decoded_data)

        return ref_data

    def get_lmbda_from_depth(self, depth: float, initial_lmbda: float) -> float:
        return initial_lmbda * (1.5**depth)

    def save(self, save_path: str) -> None:
        subprocess.call(f"mkdir -p {os.path.dirname(save_path)}", shell=True)

        self.coding_structure.unload_all_original_frames()
        self.coding_structure.unload_all_references_data()
        self.coding_structure.unload_all_decoded_data()

        data_to_save = {
            "coding_structure": self.coding_structure,
            "shared_coolchic_parameter": self.shared_coolchic_parameter,
            "shared_frame_encoder_manager": self.shared_frame_encoder_manager,
            "all_frame_encoders": {},
        }

        for k, v in self.all_frame_encoders.items():
            frame_encoder, frame_encoder_manager = v
            data_to_save["all_frame_encoders"][k] = (frame_encoder.save(), frame_encoder_manager)

        torch.save(data_to_save, save_path)

def load_video_encoder(load_path: str) -> VideoEncoder:
    print(f"Loading a video encoder from {load_path}")

    raw_data = torch.load(load_path, map_location="cpu")

    video_encoder = VideoEncoder(
        coding_structure=raw_data["coding_structure"],
        shared_coolchic_parameter=raw_data["shared_coolchic_parameter"],
        shared_frame_encoder_manager=raw_data["shared_frame_encoder_manager"],
    )

    for k, v in raw_data["all_frame_encoders"].items():
        raw_bytes_frame_encoder, frame_encoder_manager = v
        video_encoder.all_frame_encoders[k] = (
            load_frame_encoder(raw_bytes_frame_encoder),
            frame_encoder_manager,
        )

    return video_encoder
