"""Проверка реального декодирования без запуска сервера и облачных API.

Запуск: python -m unittest discover -s tests -v
Нужны зависимости проекта и FFmpeg (PATH или FFMPEG_PATH).
"""
import ast
import io
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch
import wave

from pydub import AudioSegment


ROOT = Path(__file__).resolve().parents[1]
FFMPEG = os.environ.get("FFMPEG_PATH", "").strip().strip('"') or shutil.which("ffmpeg")


def audio_functions():
    # Не импортируем main: импорт запускает серверные ресурсы и читает .env.
    tree = ast.parse((ROOT / "main.py").read_text(encoding="utf-8"))
    names = {"load_audio", "split_audio", "_transcribe_speechkit_chunk",
             "transcribe_with_speechkit", "check_ffmpeg", "process_audio"}
    module = ast.Module(body=[node for node in tree.body
                            if isinstance(node, ast.FunctionDef) and node.name in names],
                        type_ignores=[])
    namespace = {"os": os, "io": io, "subprocess": subprocess,
                 "AudioSegment": AudioSegment, "FFMPEG_PATH": FFMPEG,
                 "AUDIO_SPLIT_MS": 600_000, "TranscriptionEngine": object,
                 "time": types.SimpleNamespace(sleep=Mock()),
                 "DEEPSEEK_MODEL": "test"}
    exec(compile(module, str(ROOT / "main.py"), "exec"), namespace)
    return namespace


@unittest.skipUnless(FFMPEG, "Для проверки требуется FFmpeg")
class AudioTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.functions = audio_functions()

    def generate(self, extension, codec):
        path = self.directory / f"Новая запись 23.{extension}"
        subprocess.run([FFMPEG, "-nostdin", "-v", "error", "-y", "-f", "lavfi",
                        "-i", "sine=frequency=440:sample_rate=44100:duration=2.2",
                        "-ac", "2", "-c:a", codec, str(path)],
                       check=True, capture_output=True)
        return path

    def test_formats_and_normalized_wav(self):
        for extension, codec in [("m4a", "aac"), ("mp3", "libmp3lame"),
                                 ("wav", "pcm_s24le"), ("ogg", "libopus"),
                                 ("flac", "flac"), ("aac", "aac"),
                                 ("mp4", "aac"), ("webm", "libopus")]:
            with self.subTest(format=extension):
                path = self.generate(extension, codec)
                parts = self.functions["split_audio"](str(path))
                self.assertEqual(len(parts), 1)
                self.assertNotEqual(parts[0], str(path))
                with wave.open(parts[0], "rb") as audio:
                    self.assertEqual((audio.getnchannels(), audio.getframerate(),
                                      audio.getsampwidth()), (1, 16000, 2))
                    self.assertGreater(audio.getnframes(), 30000)

    def test_format_is_detected_from_content(self):
        path = self.generate("m4a", "aac")
        renamed = path.with_suffix(".unknown")
        path.rename(renamed)
        audio = self.functions["load_audio"](str(renamed))
        self.assertEqual((audio.channels, audio.sample_width, audio.frame_rate),
                         (1, 2, 16000))

    def test_multiple_parts_preserve_duration(self):
        path = self.generate("m4a", "aac")
        self.functions["AUDIO_SPLIT_MS"] = 1000
        parts = self.functions["split_audio"](str(path))
        self.assertEqual(len(parts), 3)
        duration = 0
        for part in parts:
            with open(part, "rb") as stream:
                duration += len(AudioSegment.from_wav(stream))
        self.assertEqual(duration, len(self.functions["load_audio"](str(path))))

    def test_invalid_file_rejected_before_api(self):
        path = self.directory / "broken.m4a"
        path.write_bytes(b"not audio")
        with self.assertRaisesRegex(ValueError, "Не удалось прочитать запись"):
            self.functions["load_audio"](str(path))

    def test_video_without_audio_rejected(self):
        path = self.directory / "silent.mp4"
        subprocess.run([FFMPEG, "-nostdin", "-v", "error", "-y", "-f", "lavfi",
                        "-i", "color=size=32x32:duration=1", "-an", str(path)],
                       check=True, capture_output=True)
        with self.assertRaisesRegex(ValueError, "не содержит аудиодорожки"):
            self.functions["load_audio"](str(path))

    def test_speechkit_payload_is_raw_mono_pcm16(self):
        post = Mock(return_value=Mock(status_code=200, json=lambda: {"result": " тест "}))
        audio = (AudioSegment.silent(duration=25000, frame_rate=48000)
                 .set_channels(2).set_sample_width(4))
        with patch.dict(sys.modules, {"requests": types.SimpleNamespace(post=post)}):
            text = self.functions["_transcribe_speechkit_chunk"](audio, "key", "folder", "ru-RU")
        self.assertEqual(text, "тест")
        request = post.call_args.kwargs
        self.assertEqual(request["params"]["format"], "lpcm")
        self.assertEqual(request["params"]["sampleRateHertz"], "16000")
        self.assertEqual(len(request["data"]), 800000)
        self.assertFalse(request["data"].startswith(b"RIFF"))
        self.assertIn("timeout", request)

    def test_speechkit_long_recording_is_chunked(self):
        calls = Mock(return_value="текст")
        self.functions["load_audio"] = Mock(return_value=AudioSegment.silent(duration=61000))
        self.functions["_transcribe_speechkit_chunk"] = calls
        with patch.dict(os.environ, {"YANDEX_API_KEY": "test", "YANDEX_FOLDER_ID": "test"}):
            text = self.functions["transcribe_with_speechkit"]("unused.m4a")
        self.assertEqual([len(call.args[0]) for call in calls.call_args_list], [25000, 25000, 11000])
        self.assertEqual(text, "текст текст текст")

    def test_ffmpeg_empty_setting_uses_path_and_invalid_executable_fails(self):
        self.functions["FFMPEG_PATH"] = ""
        with patch.object(subprocess, "run", return_value=Mock(returncode=0)) as run:
            self.assertTrue(self.functions["check_ffmpeg"]())
        self.assertEqual(run.call_args.args[0], ["ffmpeg", "-version"])
        with patch.object(subprocess, "run", return_value=Mock(returncode=1)):
            with self.assertRaises(RuntimeError):
                self.functions["check_ffmpeg"]()

    def test_parts_removed_if_transcription_fails(self):
        original = self.directory / "source.m4a"
        parts = [self.directory / f"part{i}.wav" for i in range(2)]
        for path in [original, *parts]:
            path.write_bytes(b"test")
        status = Mock()
        self.functions.update(split_audio=Mock(return_value=[str(p) for p in parts]),
                              update_task_status=status,
                              transcribe_audio=Mock(side_effect=RuntimeError("API failed")))
        self.functions["process_audio"](str(original), "file", "user", "prompt",
                                        types.SimpleNamespace(value="speechkit"))
        self.assertTrue(original.exists())
        self.assertTrue(all(not p.exists() for p in parts))
        self.assertEqual(status.call_args.kwargs["status"], "error")


class SpeechKitRetryTests(unittest.TestCase):
    def setUp(self):
        self.functions = audio_functions()
        self.audio = AudioSegment.silent(duration=25000, frame_rate=16000)

    @staticmethod
    def response(status=200, text="готово"):
        return Mock(status_code=status, text="stream timeout; request-id: test-id",
                    json=lambda: {"result": text})

    def transcribe(self, outcomes, audio=None):
        self.post = Mock(side_effect=outcomes)
        requests = types.SimpleNamespace(
            post=self.post,
            exceptions=types.SimpleNamespace(Timeout=TimeoutError, ConnectionError=ConnectionError),
        )
        with patch.dict(sys.modules, {"requests": requests}):
            return self.functions["_transcribe_speechkit_chunk"](
                self.audio if audio is None else audio, "key", "folder", "ru-RU")

    def test_504_then_success_retries_same_payload(self):
        self.assertEqual(self.transcribe([self.response(504), self.response()]), "готово")
        self.assertEqual(self.post.call_count, 2)
        self.assertEqual(self.post.call_args_list[0].kwargs["data"],
                         self.post.call_args_list[1].kwargs["data"])
        self.functions["time"].sleep.assert_called_once_with(2)

    def test_repeated_timeout_splits_without_losing_or_duplicating_samples(self):
        # Нечётное количество кадров проверяет точность границы разбиения.
        audio = AudioSegment(data=b"\x01\x00\x02\x00" * 200000 + b"\x03\x00",
                             sample_width=2, channels=1, frame_rate=16000)
        self.assertEqual(self.transcribe([self.response(504)] * 3 +
                                        [self.response(text="левая"), self.response(text="правая")],
                                        audio=audio), "левая правая")
        payloads = [call.kwargs["data"] for call in self.post.call_args_list]
        self.assertEqual(len(payloads), 5)
        self.assertEqual(payloads[3] + payloads[4], audio.raw_data)
        self.assertEqual(self.functions["time"].sleep.call_count, 2)

    def test_client_timeout_and_connection_error_retried(self):
        for error in [TimeoutError("read timed out"), ConnectionError("connection reset")]:
            with self.subTest(error=type(error).__name__):
                self.assertEqual(self.transcribe([error, self.response()]), "готово")
                self.assertEqual(self.post.call_count, 2)

    def test_transient_http_errors_retried(self):
        for status in [408, 429, 500, 502, 503]:
            with self.subTest(status=status):
                self.assertEqual(self.transcribe([self.response(status), self.response()]), "готово")
                self.assertEqual(self.post.call_count, 2)

    def test_permanent_errors_not_retried(self):
        for status in [400, 401, 403]:
            with self.subTest(status=status):
                with self.assertRaisesRegex(RuntimeError, str(status)):
                    self.transcribe([self.response(status)])
                self.assertEqual(self.post.call_count, 1)

    def test_retry_limit_preserves_request_id(self):
        with self.assertRaisesRegex(RuntimeError, "request-id: test-id"):
            self.transcribe([self.response(503)] * 3)
        self.assertEqual(self.post.call_count, 3)

    def test_short_audio_timeout_stops_without_more_splitting(self):
        with self.assertRaisesRegex(RuntimeError, "после 3 попыток"):
            self.transcribe([self.response(504)] * 3, audio=self.audio[:5000])
        self.assertEqual(self.post.call_count, 3)

    def test_all_silent_results_are_valid(self):
        self.assertEqual(self.transcribe([self.response(text="")]), "")
        self.assertEqual(self.post.call_count, 1)


if __name__ == "__main__":
    unittest.main()
