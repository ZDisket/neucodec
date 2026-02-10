from typing import Optional, Dict
import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin, ModelHubMixin, hf_hub_download

from .codec_decoder_vocos import CodecDecoderVocos


class NeuCodec(
    nn.Module,
    PyTorchModelHubMixin,
    repo_url="https://github.com/neuphonic/neucodec",
    license="apache-2.0",
):

    def __init__(self, sample_rate: int, hop_length: int):
        super().__init__()
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.generator = CodecDecoderVocos(hop_length=hop_length)
        self.fc_post_a = nn.Linear(2048, 1024)

    @property
    def device(self):
        return next(self.parameters()).device

    @classmethod
    def _from_pretrained(
        cls,
        *,
        model_id: str,
        revision: Optional[str] = None,
        cache_dir: Optional[str] = None,
        force_download: bool = False,
        proxies: Optional[Dict] = None,
        resume_download: bool = False,
        local_files_only: bool = False,
        token: Optional[str] = None,
        map_location: str = "cpu",
        strict: bool = True,
        **model_kwargs,
    ):
        
        assert model_id in ["neuphonic/neucodec", "neuphonic/distill-neucodec"]
        if model_id == "neuphonic/neucodec": 
            ignore_keys = ["fc_post_s", "SemanticDecoder"]
        elif model_id == "neuphonic/distill-neucodec":
            ignore_keys = []

        # download the model weights file
        ckpt_path = hf_hub_download(
            repo_id=model_id,
            filename="pytorch_model.bin",
            revision=revision,
            cache_dir=cache_dir,
            force_download=force_download,
            proxies=proxies,
            resume_download=resume_download,
            local_files_only=local_files_only,
            token=token,
        )

        # download meta.yaml to track number of downloads
        _ = hf_hub_download(
            repo_id=model_id,
            filename="meta.yaml",
            revision=revision,
            cache_dir=cache_dir,
            force_download=force_download,
            proxies=proxies,
            resume_download=resume_download,
            local_files_only=local_files_only,
            token=token,
        )

        # initialize model
        model = cls(24_000, 480)

        # load weights
        state_dict = torch.load(ckpt_path, map_location)
        contains_list = lambda s, l: any(i in s for i in l)
        state_dict = {
            k:v for k, v in state_dict.items() 
            if not contains_list(k, ignore_keys)
        }

        # TODO: we can move to strict loading once we clean up the checkpoints
        model.load_state_dict(state_dict, strict=False)

        return model

    def decode_code(self, fsq_codes: torch.Tensor) -> torch.Tensor:
        """
        Args:
            fsq_codes: torch.Tensor [B, 1, F], 50hz FSQ codes

        Returns:
            recon: torch.Tensor [B, 1, T], reconstructed 24kHz audio
        """

        fsq_post_emb = self.generator.quantizer.get_output_from_indices(fsq_codes.transpose(1, 2))
        fsq_post_emb = fsq_post_emb.transpose(1, 2)
        fsq_post_emb = self.fc_post_a(fsq_post_emb.transpose(1, 2)).transpose(1, 2) 
        recon = self.generator(fsq_post_emb.transpose(1, 2), vq=False)[0]
        return recon
    

class DistillNeuCodec(NeuCodec):
    def __init__(self, sample_rate: int, hop_length: int):
        nn.Module.__init__(self)
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.generator = CodecDecoderVocos(hop_length=hop_length)
        self.fc_post_a = nn.Linear(2048, 1024)


class NeuCodecOnnxDecoder(
    ModelHubMixin,
    repo_url="https://github.com/neuphonic/neucodec",
    license="apache-2.0",
):
    
    def __init__(self, onnx_path):
        
        # onnx import
        try: 
            import onnxruntime
        except ImportError as e:
            raise ImportError("Failed to import `onnxruntime`. Install with the following command: pip install onnxruntime") from e
        
        # load model
        so = onnxruntime.SessionOptions()
        so.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = onnxruntime.InferenceSession(
            onnx_path,
            sess_options=so
        )
        self.sample_rate = 24_000

    @classmethod
    def _from_pretrained(
        cls,
        *,
        model_id: str,
        revision: Optional[str] = None,
        cache_dir: Optional[str] = None,
        force_download: bool = False,
        proxies: Optional[Dict] = None,
        resume_download: bool = False,
        local_files_only: bool = False,
        token: Optional[str] = None,
        map_location: str = "cpu",
        strict: bool = True,
        **model_kwargs,
    ):
        
        # download the model weights file
        onnx_path = hf_hub_download(
            repo_id=model_id,
            filename="model.onnx",
            revision=revision,
            cache_dir=cache_dir,
            force_download=force_download,
            proxies=proxies,
            resume_download=resume_download,
            local_files_only=local_files_only,
            token=token,
        )

        # download meta.yaml to track number of downloads
        _ = hf_hub_download(
            repo_id=model_id,
            filename="meta.yaml",
            revision=revision,
            cache_dir=cache_dir,
            force_download=force_download,
            proxies=proxies,
            resume_download=resume_download,
            local_files_only=local_files_only,
            token=token,
        )

        # initialize model
        model = cls(onnx_path)

        # only support CPU
        if map_location != "cpu":
            raise ValueError("The onnx decoder currently only supports CPU runtimes.")

        return model
    
    def decode_code(self, codes: np.ndarray) -> np.ndarray:
        """
        Args:
            fsq_codes: np.array [B, 1, F], 50hz FSQ codes

        Returns:
            recon: np.array [B, 1, T], reconstructed 24kHz audio
        """

        # validate inputs
        if not isinstance(codes, np.ndarray):
            raise ValueError("`Codes` should be an np.array.")
        if not len(codes.shape) == 3 or codes.shape[1] != 1:
            raise ValueError("`Codes` should be of shape [B, 1, F].")

        # run decoder
        recon = self.session.run(
            None, {"codes": codes}
        )[0].astype(np.float32)
        
        return recon
