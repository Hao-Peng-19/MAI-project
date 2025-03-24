from pydub import AudioSegment


def extend_wav(input_wav, output_wav, target_duration_ms):
    # 读取音频文件
    audio = AudioSegment.from_wav(input_wav)

    # 计算需要填充的时长
    silence_duration = max(0, target_duration_ms - len(audio))

    # 仅当音频长度不足目标时长时才填充静音
    if silence_duration > 0:
        silence = AudioSegment.silent(duration=silence_duration)
        audio += silence  # 在末尾添加静音

    # 保存新的 WAV 文件
    audio.export(output_wav, format="wav")
    print(f"音频已延长至 {target_duration_ms / 1000} 秒，并保存到 {output_wav}")


# 示例用法
extend_wav("cat.wav", "cat-o.wav", 5500)  # 目标时长 5000 毫秒（5 秒）
