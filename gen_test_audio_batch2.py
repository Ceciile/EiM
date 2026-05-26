# gen_test_audio_batch2.py
# 追加到已有 test_audio/ 目录，与 batch1 共用同一 ground_truth.json

import json
import os
from pathlib import Path

from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs

from asr_spike import ENV_FILE, load_env_file

load_env_file(ENV_FILE)
_api_key = os.environ.get("ELEVENLABS_API_KEY")
if not _api_key:
    raise RuntimeError("Missing ELEVENLABS_API_KEY in env or environment")
client = ElevenLabs(api_key=_api_key)

OUTPUT_DIR = Path("test_audio")
OUTPUT_DIR.mkdir(exist_ok=True)
GT_FILE = OUTPUT_DIR / "ground_truth.json"

# ── 新增脚本：印度男 + 泰国女，60/40 英中 ─────────────────
SCRIPTS_BATCH2 = {

    # ── 印度男 · 吃饭场景 ──────────────────────────────────
    "E_in_meal_full": {
        "voice_id": "ErXwobaYiN019PkySvjV",   # Antoni — 较中性男声，Indian accent tag 效果稳
        "text": (
            "[Indian accent] Teacher, today I went to the 食堂 for lunch. "
            "I wanted to order 炒饭 but I said 抄饭 by mistake — "
            "the 阿姨 looked so confused at me only. "
            "Then my classmate helped, he said 我朋友想要炒饭. "
            "After eating we also had 汤, very nice. "
            "I think my 发音 is still very bad, "
            "can you teach me 食堂 and 炒饭 one more time please?"
        ),
        "note": "印度男·吃饭场景·完整句·60/40英中",
        "gt": (
            "Teacher, today I went to the 食堂 for lunch. "
            "I wanted to order 炒饭 but I said 抄饭 by mistake — "
            "the 阿姨 looked so confused at me only. "
            "Then my classmate helped, he said 我朋友想要炒饭. "
            "After eating we also had 汤, very nice. "
            "I think my 发音 is still very bad, "
            "can you teach me 食堂 and 炒饭 one more time please?"
        ),
    },

    "F_in_meal_missing": {
        "voice_id": "ErXwobaYiN019PkySvjV",
        "text": (
            # 刻意缺开头半句，从解释声调开始
            "[Indian accent] 四声 is confusing me only. "
            "Like 吃 and 七, they sound same to me but meaning is totally different. "
            "Yesterday I told my friend 我想吃饭 "
            "but he thought I said something else and laughed. "
            "这个 tone practice, how many times I must do per day?"
        ),
        "note": "印度男·吃饭场景·缺开头半句",
        "gt": (
            "四声 is confusing me only. "
            "Like 吃 and 七, they sound same to me but meaning is totally different. "
            "Yesterday I told my friend 我想吃饭 "
            "but he thought I said something else and laughed. "
            "这个 tone practice, how many times I must do per day?"
        ),
    },

    # ── 泰国女 · 上学场景 ──────────────────────────────────
    "G_th_school_full": {
        "voice_id": "EXAVITQu4vr4xnSDxMaL",  # Bella — 女声，Thai accent tag 效果好
        "text": (
            "[Thai accent] 老师好！Today I come to 学校 very early, "
            "maybe 七点半. I study 中文 every morning before class. "
            "But I have problem with 上课 time — "
            "I always late because I cannot find the 教室. "
            "My friend say turn left at 图书馆, "
            "then go straight, but I still get lost. "
            "Can you write down the 教室号码 for me? 谢谢老师！"
        ),
        "note": "泰国女·上学场景·完整句·60/40英中",
        "gt": (
            "老师好！Today I come to 学校 very early, "
            "maybe 七点半. I study 中文 every morning before class. "
            "But I have problem with 上课 time — "
            "I always late because I cannot find the 教室. "
            "My friend say turn left at 图书馆, "
            "then go straight, but I still get lost. "
            "Can you write down the 教室号码 for me? 谢谢老师！"
        ),
    },

    "H_th_school_missing": {
        "voice_id": "EXAVITQu4vr4xnSDxMaL",
        "text": (
            # 缺开头，从困惑处开始
            "[Thai accent] 作业 is too much for me. "
            "I have 三个 assignments this week, "
            "plus the 口语 test on Friday. "
            "I try to practice with my 同学 but she speak 普通话 too fast. "
            "When I say 我不知道, she always laugh — "
            "maybe my 声调 is very funny? 对不起 teacher, "
            "I will try harder next week."
        ),
        "note": "泰国女·上学场景·缺开头半句",
        "gt": (
            "作业 is too much for me. "
            "I have 三个 assignments this week, "
            "plus the 口语 test on Friday. "
            "I try to practice with my 同学 but she speak 普通话 too fast. "
            "When I say 我不知道, she always laugh — "
            "maybe my 声调 is very funny? 对不起 teacher, "
            "I will try harder next week."
        ),
    },
}

# ── Voice settings 按口音特点微调 ─────────────────────────
VOICE_SETTINGS = {
    "indian": VoiceSettings(
        stability=0.40,        # 低 stability → 卷舌/节奏感更明显
        similarity_boost=0.70,
        style=0.35,
        use_speaker_boost=True,
    ),
    "thai": VoiceSettings(
        stability=0.50,        # 泰国口音较平，稍高 stability 保持音调特征
        similarity_boost=0.72,
        style=0.25,
        use_speaker_boost=True,
    ),
}

def generate(key: str, cfg: dict):
    out_path = OUTPUT_DIR / f"{key}.mp3"
    if out_path.exists():
        print(f"  [skip] {key}.mp3")
        return

    is_indian = "_in_" in key
    settings = VOICE_SETTINGS["indian"] if is_indian else VOICE_SETTINGS["thai"]

    print(f"  Generating {key} — {cfg['note']}")
    audio = client.text_to_speech.convert(
        voice_id=cfg["voice_id"],
        model_id="eleven_v3",
        text=cfg["text"],
        voice_settings=settings,
        output_format="mp3_44100_128",
    )
    with open(out_path, "wb") as f:
        for chunk in audio:
            f.write(chunk)
    print(f"  ✓ {out_path}  ({out_path.stat().st_size//1024} KB)")


def main():
    print("=== Batch 2：印度男 + 泰国女 ===\n")

    # 读取已有 GT（batch1），追加 batch2
    existing = {}
    if GT_FILE.exists():
        existing = json.load(open(GT_FILE))

    for key, cfg in SCRIPTS_BATCH2.items():
        generate(key, cfg)
        existing[key] = {
            "file": str(OUTPUT_DIR / f"{key}.mp3"),
            "note": cfg["note"],
            "gt":   cfg["gt"],
        }

    with open(GT_FILE, "w", ensure_ascii=False) as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)

    print(f"\n✓ ground_truth.json 已更新，共 {len(existing)} 条")
    print("\n所有音频文件：")
    for p in sorted(OUTPUT_DIR.glob("*.mp3")):
        print(f"  {p.name:<30} {p.stat().st_size//1024:>5} KB")


if __name__ == "__main__":
    main()