# pip install elevenlabs openai python-dotenv

import os
from pathlib import Path
from elevenlabs.client import ElevenLabs
from elevenlabs import VoiceSettings

client = ElevenLabs(api_key=os.environ["ELEVENLABS_API_KEY"])

# ── 测试文本设计 ─────────────────────────────────────────
# 模拟东南亚学员上中文课时的真实交互：
# 主要说带口音的英语，夹带少量有口音的中文短语
SCRIPTS = {
    "A_sg_full": {
        "text": "[Singaporean accent] Teacher, I don't understand lah. "
                "Can you repeat the 声调 again? I always mix up the second tone "
                "and the third tone one. Like 买 and 卖, I cannot tell the difference.",
        "note": "新加坡口音，完整句，含声调相关中文词",
        "gt":   "Teacher, I don't understand lah. Can you repeat the 声调 again? "
                "I always mix up the second tone and the third tone one. "
                "Like 买 and 卖, I cannot tell the difference."
    },
    "B_sg_full": {
        "text": "[Singaporean accent] Okay I try. 你好，我叫 Sarah. "
                "My 普通话 is not very good but I am trying my best, "
                "can or not?",
        "note": "新加坡口音，自我介绍混说场景",
        "gt":   "Okay I try. 你好，我叫 Sarah. My 普通话 is not very good "
                "but I am trying my best, can or not?"
    },
    "C_ph_full": {
        "text": "[Filipino accent] Hello teacher! Yesterday I practice saying "
                "四 and 十, but my roommate said I sound very funny. "
                "How to make the si sound correctly? Is it like this — 四四四?",
        "note": "菲律宾口音，完整句，含数字发音问题",
        "gt":   "Hello teacher! Yesterday I practice saying 四 and 十, "
                "but my roommate said I sound very funny. "
                "How to make the si sound correctly? Is it like this — 四四四?"
    },
    "D_ph_missing": {
        # 刻意从中间开始，模拟文件D"缺开头半句"
        "text": "[Filipino accent] my 妈妈 always say I need to study more. "
                "She want me to learn Chinese because for work is very useful. "
                "So now I am here learning, 加油 right?",
        "note": "菲律宾口音，缺开头半句，从中间进入",
        "gt":   "my 妈妈 always say I need to study more. "
                "She want me to learn Chinese because for work is very useful. "
                "So now I am here learning, 加油 right?"
    },
}

OUTPUT_DIR = Path("test_audio")
OUTPUT_DIR.mkdir(exist_ok=True)

GT_FILE = OUTPUT_DIR / "ground_truth.json"

import json

def generate_audio(key: str, cfg: dict):
    out_path = OUTPUT_DIR / f"{key}.mp3"
    if out_path.exists():
        print(f"  [skip] {key}.mp3 already exists")
        return

    print(f"  Generating {key}: {cfg['note']}")
    audio = client.text_to_speech.convert(
        voice_id="pNInz6obpgDQGcFmaJgB",  # Adam — 中性男声，口音 tag 效果稳定
        model_id="eleven_v3",              # v3 才支持 audio tags
        text=cfg["text"],
        voice_settings=VoiceSettings(
            stability=0.45,        # 稍低 stability 让口音更自然
            similarity_boost=0.75,
            style=0.3,
            use_speaker_boost=True,
        ),
        output_format="mp3_44100_128",
    )
    with open(out_path, "wb") as f:
        for chunk in audio:
            f.write(chunk)
    print(f"  ✓ Saved: {out_path}")


def main():
    print("=== 生成测试音频 ===\n")

    gt_data = {}
    for key, cfg in SCRIPTS.items():
        generate_audio(key, cfg)
        gt_data[key] = {
            "file":  str(OUTPUT_DIR / f"{key}.mp3"),
            "note":  cfg["note"],
            "gt":    cfg["gt"],
        }

    with open(GT_FILE, "w", ensure_ascii=False) as f:
        json.dump(gt_data, f, ensure_ascii=False, indent=2)
    print(f"\n✓ Ground truth 已保存: {GT_FILE}")
    print("\n生成文件列表:")
    for p in sorted(OUTPUT_DIR.glob("*.mp3")):
        size = p.stat().st_size / 1024
        print(f"  {p.name:<25} {size:>6.1f} KB")


if __name__ == "__main__":
    main()