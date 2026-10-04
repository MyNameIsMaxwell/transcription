// Распознавание голосовых через Yandex SpeechKit (синхронный REST v1).
// Браузер присылает сырой PCM: 16 кГц, моно, 16 бит (little-endian).
// v1 принимает до 30 с / 1 МБ за запрос, поэтому режем на куски по 25 с.

import { checkRateLimit, clientIp } from "@/lib/rate-limit";

export const maxDuration = 60;

const SAMPLE_RATE = 16000;
const BYTES_PER_SEC = SAMPLE_RATE * 2;
const CHUNK_BYTES = BYTES_PER_SEC * 25;
const MAX_SECONDS = 5 * 60;
const PARALLEL = 4;

async function recognizeChunk(pcm: ArrayBuffer): Promise<string> {
  const params = new URLSearchParams({
    folderId: process.env.YANDEX_FOLDER_ID ?? "",
    lang: process.env.YANDEX_LANG || "ru-RU",
    format: "lpcm",
    sampleRateHertz: String(SAMPLE_RATE),
    topic: "general",
  });
  const res = await fetch(`https://stt.api.cloud.yandex.net/speech/v1/stt:recognize?${params}`, {
    method: "POST",
    headers: { Authorization: `Api-Key ${process.env.YANDEX_API_KEY ?? ""}` },
    body: pcm,
    signal: AbortSignal.timeout(30000),
  });
  if (!res.ok) throw new Error(`SpeechKit ${res.status}: ${await res.text()}`);
  const json = (await res.json()) as { result?: string };
  return json.result ?? "";
}

export async function POST(req: Request) {
  const limited = checkRateLimit(
    "voice",
    req.headers.get("x-conversation-id") ?? "unknown",
    clientIp(req),
  );
  if (limited) return Response.json({ error: limited }, { status: 429 });

  const pcm = await req.arrayBuffer();
  if (pcm.byteLength < BYTES_PER_SEC / 4) {
    return Response.json({ error: "Слишком короткая запись" }, { status: 400 });
  }
  if (pcm.byteLength > BYTES_PER_SEC * MAX_SECONDS) {
    return Response.json({ error: "Голосовое длиннее 5 минут — разбейте на несколько" }, { status: 413 });
  }

  const chunks: ArrayBuffer[] = [];
  for (let i = 0; i < pcm.byteLength; i += CHUNK_BYTES) chunks.push(pcm.slice(i, i + CHUNK_BYTES));

  try {
    const texts: string[] = new Array(chunks.length);
    for (let i = 0; i < chunks.length; i += PARALLEL) {
      const batch = chunks.slice(i, i + PARALLEL);
      const out = await Promise.all(batch.map(recognizeChunk));
      out.forEach((t, j) => (texts[i + j] = t));
    }
    const text = texts.filter(Boolean).join(" ").trim();
    return Response.json({ text });
  } catch (e) {
    console.error(e);
    return Response.json({ error: "Не удалось распознать голосовое" }, { status: 502 });
  }
}
