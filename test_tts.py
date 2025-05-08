from melo.api import TTS
import os
import torch
texts = {
    'EN_NEWEST': "Did you ever hear a folk tale about a giant turtle?",  # The newest English base speaker model
    # 'EN': "Did you ever hear a folk tale about a giant turtle?",
    # 'ES': "El resplandor del sol acaricia las olas, pintando el cielo con una paleta deslumbrante.",
    # 'FR': "La lueur dorée du soleil caresse les vagues, peignant le ciel d'une palette éblouissante.",
    # 'ZH': "在这次vacation中，我们计划去Paris欣赏埃菲尔铁塔和卢浮宫的美景。",
    # 'JP': "彼は毎朝ジョギングをして体を健康に保っています。",
    # 'KR': "안녕하세요! 오늘은 날씨가 정말 좋네요.",
}

pt_device = 'cpu'
output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)
# Speed is adjustable
speed = 1.0

language, text =  texts[0]
model = TTS(language=language, device=pt_device)
speaker_ids = model.hps.data.spk2id
    
for speaker_key in speaker_ids.keys():
    speaker_id = speaker_ids[speaker_key]
    speaker_key = speaker_key.lower().replace('_', '-')
    save_path = f'{output_dir}/output_v2_{speaker_key}.wav'
    model.tts_to_file(text, speaker_id, save_path, speed=speed)



