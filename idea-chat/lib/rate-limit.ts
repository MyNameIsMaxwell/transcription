// Простой лимит запросов в памяти процесса (одного инстанса Next.js достаточно).
// Два уровня:
//  - на диалог: RATE_LIMIT_PER_CHAT запросов за окно — защищает от «залипших» и бесконечных диалогов;
//  - на IP: RATE_LIMIT_PER_IP за окно — от спама. Он заметно выше, потому что весь завод
//    может выходить в интернет с одного внешнего IP.

const WINDOW_MS = Number(process.env.RATE_LIMIT_WINDOW_MIN || 10) * 60_000;
const PER_CHAT = Number(process.env.RATE_LIMIT_PER_CHAT || 20);
const PER_IP = Number(process.env.RATE_LIMIT_PER_IP || 200);

const hits = new Map<string, number[]>();

function hit(key: string, limit: number) {
  const now = Date.now();
  const list = (hits.get(key) ?? []).filter((t) => now - t < WINDOW_MS);
  if (list.length >= limit) {
    hits.set(key, list);
    return Math.ceil((WINDOW_MS - (now - list[0])) / 60_000);
  }
  list.push(now);
  hits.set(key, list);
  return 0;
}

export function clientIp(req: Request) {
  return (
    req.headers.get("x-forwarded-for")?.split(",")[0]?.trim() ||
    req.headers.get("x-real-ip") ||
    "local"
  );
}

/** Возвращает текст ошибки, если лимит превышен, иначе null. */
export function checkRateLimit(kind: string, conversationId: string, ip: string): string | null {
  const waitIp = hit(`${kind}:ip:${ip}`, PER_IP);
  const waitChat = waitIp ? 0 : hit(`${kind}:chat:${conversationId}`, PER_CHAT);
  const wait = waitIp || waitChat;
  if (!wait) return null;
  return `Слишком много сообщений подряд. Подождите примерно ${wait} мин. и продолжите — переписка сохранится.`;
}
