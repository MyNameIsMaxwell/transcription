"use client";

import { useAui, useAuiState } from "@assistant-ui/react";
import { Loader2Icon, MicIcon, SendHorizontalIcon, XIcon } from "lucide-react";
import { useEffect, useRef, useState, type FC } from "react";
import { getConversationId } from "@/lib/conversation-id";
import { TooltipIconButton } from "@/components/assistant-ui/elements/tooltip-icon-button";

const TARGET_RATE = 16000;
const MAX_SECONDS = 5 * 60;

/** Любая запись браузера → PCM 16 кГц моно 16 бит (формат lpcm для SpeechKit). */
async function toPcm16(blob: Blob): Promise<ArrayBuffer> {
  const ctx = new AudioContext();
  try {
    const decoded = await ctx.decodeAudioData(await blob.arrayBuffer());
    const offline = new OfflineAudioContext(1, Math.ceil(decoded.duration * TARGET_RATE), TARGET_RATE);
    const src = offline.createBufferSource();
    src.buffer = decoded;
    src.connect(offline.destination);
    src.start();
    const rendered = await offline.startRendering();
    const f32 = rendered.getChannelData(0);
    const i16 = new Int16Array(f32.length);
    for (let i = 0; i < f32.length; i++) {
      const s = Math.max(-1, Math.min(1, f32[i]));
      i16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
    }
    return i16.buffer;
  } finally {
    void ctx.close();
  }
}

const fmt = (s: number) => `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;

type State = "idle" | "recording" | "processing";

/** Кнопка голосового сообщения «как в Телеграме»: записал → отправил → распознанный текст уходит в чат. */
export const VoiceRecorder: FC = () => {
  const aui = useAui();
  // onstop-обработчик создаётся в момент старта записи — текущий текст читаем через ref.
  const composerTextRef = useRef("");
  composerTextRef.current = useAuiState((s) => s.composer.text);
  const isRunning = useAuiState((s) => s.thread.isRunning);
  const [state, setState] = useState<State>("idle");
  const [seconds, setSeconds] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const cancelledRef = useRef(false);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => () => stopTracks(), []);

  function stopTracks() {
    if (timerRef.current) clearInterval(timerRef.current);
    recorderRef.current?.stream.getTracks().forEach((t) => t.stop());
  }

  async function start() {
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const rec = new MediaRecorder(stream);
      chunksRef.current = [];
      cancelledRef.current = false;
      rec.ondataavailable = (e) => e.data.size && chunksRef.current.push(e.data);
      rec.onstop = () => void finish(rec.mimeType);
      rec.start();
      recorderRef.current = rec;
      setSeconds(0);
      setState("recording");
      timerRef.current = setInterval(() => {
        setSeconds((s) => {
          if (s + 1 >= MAX_SECONDS) recorderRef.current?.stop();
          return s + 1;
        });
      }, 1000);
    } catch {
      setError("Нет доступа к микрофону");
    }
  }

  function stop(cancel: boolean) {
    cancelledRef.current = cancel;
    recorderRef.current?.stop();
  }

  async function finish(mimeType: string) {
    stopTracks();
    if (cancelledRef.current) {
      setState("idle");
      return;
    }
    setState("processing");
    try {
      const pcm = await toPcm16(new Blob(chunksRef.current, { type: mimeType }));
      const res = await fetch("/api/transcribe", {
        method: "POST",
        headers: {
          "Content-Type": "application/octet-stream",
          "x-conversation-id": getConversationId(),
        },
        body: pcm,
      });
      const json = (await res.json()) as { text?: string; error?: string };
      if (!res.ok) throw new Error(json.error || "Ошибка распознавания");
      if (!json.text) throw new Error("Не удалось разобрать слова — попробуйте ещё раз");
      const typed = composerTextRef.current.trim();
      aui.composer.setText(`${typed ? typed + "\n" : ""}🎤 ${json.text}`);
      aui.composer.send();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Ошибка распознавания");
    } finally {
      setState("idle");
    }
  }

  if (state === "recording") {
    return (
      <div className="flex items-center gap-1.5">
        <span className="text-destructive flex items-center gap-1.5 text-sm tabular-nums">
          <span className="bg-destructive size-2 animate-pulse rounded-full" />
          {fmt(seconds)}
        </span>
        <TooltipIconButton
          tooltip="Отменить"
          side="bottom"
          type="button"
          variant="ghost"
          size="icon"
          className="text-muted-foreground size-7 rounded-full"
          onClick={() => stop(true)}
        >
          <XIcon className="size-4" />
        </TooltipIconButton>
        <TooltipIconButton
          tooltip="Отправить голосовое"
          side="bottom"
          type="button"
          variant="default"
          size="icon"
          className="size-7 rounded-full"
          onClick={() => stop(false)}
        >
          <SendHorizontalIcon className="size-4" />
        </TooltipIconButton>
      </div>
    );
  }

  return (
    <div className="flex items-center gap-1.5">
      {error && <span className="text-destructive max-w-48 truncate text-xs">{error}</span>}
      <TooltipIconButton
        tooltip={state === "processing" ? "Распознаю…" : "Записать голосовое"}
        side="bottom"
        type="button"
        variant="ghost"
        size="icon"
        className="text-muted-foreground hover:text-foreground size-7 rounded-full"
        disabled={state === "processing" || isRunning}
        onClick={start}
      >
        {state === "processing" ? (
          <Loader2Icon className="size-4 animate-spin" />
        ) : (
          <MicIcon className="size-4" />
        )}
      </TooltipIconButton>
    </div>
  );
};
