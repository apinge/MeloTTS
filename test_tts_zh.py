from melo.api import TTS
import os
import torch
from pathlib import Path
# texts = {
#     'EN_NEWEST': "Did you ever hear a folk tale about a fox spirit?",  # The newest English base speaker model
#     # 'EN': "Did you ever hear a folk tale about a giant turtle?",
#     # 'ES': "El resplandor del sol acaricia las olas, pintando el cielo con una paleta deslumbrante.",
#     # 'FR': "La lueur dorée du soleil caresse les vagues, peignant le ciel d'une palette éblouissante.",
#     # 'ZH': "在这次vacation中，我们计划去Paris欣赏埃菲尔铁塔和卢浮宫的美景。",
#     # 'JP': "彼は毎朝ジョギングをして体を健康に保っています。",
#     # 'KR': "안녕하세요! 오늘은 날씨가 정말 좋네요.",
# }

pt_device = 'cpu'
output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)
# Speed is adjustable
speed = 1.0
language = 'ZH'
text = "在这次vacation中，我们计划去Paris欣赏埃菲尔铁塔和卢浮宫的美景。"
model = TTS(language=language, device='cpu')
speaker_ids = model.hps.data.spk2id
    
# for speaker_key in speaker_ids.keys():
#     speaker_id = speaker_ids[speaker_key]
#     speaker_key = speaker_key.lower().replace('_', '-')
#     save_path = f'{output_dir}/output_v2_{speaker_key}.wav'
#     model.tts_to_file(text, speaker_id, save_path, speed=speed)
   
"""
Convert the TTS model to OpenVINO IR 
"""


from melo.api import SynthesizerTrn
from melo.download_utils  import load_or_download_config, load_or_download_model
import torch.nn as nn

class SynthesizerTTSWrapper(nn.Module):
    """
    Wrapper for SynthesizerTrn model to make it compatible with Torch-style inference.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, x_lengths, sid, tone, language, bert, ja_bert,noise_scale, length_scale, noise_scale_w, sdp_ratio):
        """
        Forward call to the underlying SynthesizerTrn model. Accepts arbitrary arguments
        and forwards them directly to the model's inference method.
        """
        return self.model.infer( 
                        x,
                        x_lengths,
                        sid,
                        tone,
                        language,
                        bert,
                        ja_bert,
                        sdp_ratio = sdp_ratio,
                        noise_scale = noise_scale,
                        noise_scale_w = noise_scale_w,
                        length_scale = length_scale)

    def get_example_input(self):
        """
        Return a tuple of example inputs for tracing/ONNX exporting or debugging.
        """
        x_tst = torch.tensor([[  0,   0,   0, 100,   0,  13,   0, 101,   0,  26,   0,  21,   0,  41,
           0,   7,   0,  33,   0,  57,   0,  33,   0,  77,   0,  12,   0,  62,
           0, 101,   0,  67,   0, 106,   0,   0,   0]],dtype=torch.int64).to(pt_device)

        x_tst_lengths = torch.tensor([37],dtype=torch.int64).to(pt_device)
        speakers = torch.tensor([1],dtype=torch.int64).to(pt_device)
        tones = torch.tensor([[0, 0, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 7, 0, 8, 0, 7, 0, 9, 0, 7,
         0, 8, 0, 7, 0, 1, 0, 1, 0, 0, 0, 0, 0]],dtype=torch.int64).to(pt_device)
        lang_ids = torch.tensor([[0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3,
         0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0]],dtype=torch.int64).to(pt_device)
        bert = torch.zeros((1, 1024, 37), dtype=torch.float32).to(pt_device)
        ja_bert = torch.randn(1, 768, 37).float().to(pt_device)
        sdp_ratio = torch.tensor(0.2).to(pt_device)
        noise_scale = torch.tensor(0.6).to(pt_device)
        noise_scale_w = torch.tensor(0.8).to(pt_device)
        length_scale = torch.tensor(1.0).to(pt_device)

        return (
            x_tst,
            x_tst_lengths,
            speakers,
            tones,
            lang_ids,
            bert,
            ja_bert,
            noise_scale,
            length_scale,
            noise_scale_w,
            sdp_ratio,
        )
    
    # config_path = 
"""
Test OpenVINO IR
"""
import openvino as ov
tts_model = SynthesizerTTSWrapper(model.model)

EN_TTS_IR = Path("/home/gta/qiu/MeloTTS/openvino_irs/openvoice_zh_tts.xml")

if not EN_TTS_IR.exists():
    ov_model = ov.convert_model(tts_model, example_input=tts_model.get_example_input())
    ov.save_model(ov_model, EN_TTS_IR )
core = ov.Core()
def get_pathched_infer(ov_model: ov.Model, device: str) -> callable:
    compiled_model = core.compile_model(ov_model, device)
    def infer_impl(x, x_lengths, sid, tone, language, bert, ja_bert,noise_scale, length_scale, noise_scale_w, max_len = None, sdp_ratio=1.0,y = None, g = None):
        ov_output = compiled_model((x, x_lengths, sid, tone, language, bert, ja_bert,noise_scale, length_scale, noise_scale_w, sdp_ratio))
        return (torch.tensor(ov_output[0]),)
    return infer_impl


model.model.infer = get_pathched_infer(EN_TTS_IR, "CPU")

speaker_ids = model.hps.data.spk2id
    
for speaker_key in speaker_ids.keys():
    speaker_id = speaker_ids[speaker_key]
    speaker_key = speaker_key.lower().replace('_', '-')
    save_path = f'{output_dir}/output_ov_{speaker_key}.wav'
    model.tts_to_file(text, speaker_id, save_path, speed=speed)