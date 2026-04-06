# Copyright (c) Ruopeng Gao. All Rights Reserved.

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class MOTIP(nn.Module):
    def __init__(
            self,
            detr: nn.Module,
            detr_framework: str,
            only_detr: bool,
            trajectory_modeling: nn.Module,
            id_decoder: nn.Module,
    ):
        super().__init__()
        self.detr = detr
        self.detr_framework = detr_framework
        self.only_detr = only_detr
        self.trajectory_modeling = trajectory_modeling
        self.id_decoder = id_decoder

        self.flow_rgb_adapter = nn.Conv2d(6, 3, kernel_size=1, bias=False)
        with torch.no_grad():
            self.flow_rgb_adapter.weight.zero_()
            for c in range(3):
                self.flow_rgb_adapter.weight[c, c, 0, 0] = 1.0

        if self.id_decoder is not None:
            self.num_id_vocabulary = self.id_decoder.num_id_vocabulary
        else:
            self.num_id_vocabulary = 1000           # hack implementation

        return

    def forward(self, **kwargs):
        assert "part" in kwargs, "Parameter `part` is required for MOTIP forward."
        match kwargs["part"]:
            case "detr":
                frames = kwargs["frames"]
                if frames.tensors.shape[1] == 6:
                    frames = frames.clone()
                    frames.tensors = self.flow_rgb_adapter(frames.tensors)
                if "use_checkpoint" in kwargs:
                    return checkpoint(
                        self.detr, frames,
                        use_reentrant=False,
                    )
                else:
                    return self.detr(samples=frames)
            case "trajectory_modeling":
                seq_info = kwargs["seq_info"]
                return self.trajectory_modeling(seq_info)
            case "id_decoder":
                seq_info = kwargs["seq_info"]
                use_decoder_checkpoint = kwargs["use_decoder_checkpoint"] if "use_decoder_checkpoint" in kwargs else False
                return self.id_decoder(seq_info, use_decoder_checkpoint=use_decoder_checkpoint)
            case _:
                raise NotImplementedError(f"MOTIP forwarding doesn't support part={kwargs['part']}.")
