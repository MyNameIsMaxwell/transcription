// Хранилище на SQLite (встроенный в Node 22+ модуль node:sqlite, без нативных зависимостей).
// Файл базы: data/ideas.db. Идеи — отдельные колонки по шаблону + полный JSON;
// диалоги — JSON-массив сообщений (формат UIMessage из AI SDK).
import { DatabaseSync } from "node:sqlite";
import { mkdirSync } from "node:fs";
import path from "node:path";
import type { UIMessage } from "ai";
import { IDEA_FIELDS, type Card, type Idea } from "./idea-template";

const DATA_DIR = path.join(process.cwd(), "data");
const FIELD_KEYS = IDEA_FIELDS.map((f) => f.key);

export type IdeaRow = Idea & {
  id: number;
  created_at: string;
  conversation_id: string;
  anonymous: number;
  status: string;
};

let db: DatabaseSync | undefined;

function getDb() {
  if (db) return db;
  mkdirSync(DATA_DIR, { recursive: true });
  db = new DatabaseSync(path.join(DATA_DIR, "ideas.db"));
  db.exec(`
    PRAGMA journal_mode = WAL;
    CREATE TABLE IF NOT EXISTS conversations (
      id TEXT PRIMARY KEY,
      created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
      updated_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
      ip TEXT,
      message_count INTEGER NOT NULL DEFAULT 0,
      messages_json TEXT NOT NULL DEFAULT '[]',
      idea_id INTEGER
    );
    CREATE TABLE IF NOT EXISTS ideas (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
      conversation_id TEXT NOT NULL,
      anonymous INTEGER NOT NULL DEFAULT 0,
      status TEXT NOT NULL DEFAULT 'новая',
      ${FIELD_KEYS.map((k) => `${k} TEXT`).join(",\n      ")},
      dialog_text TEXT
    );
  `);
  const convCols = new Set(
    (db.prepare("PRAGMA table_info(conversations)").all() as { name: string }[]).map((c) => c.name),
  );
  if (!convCols.has("card_json")) db.exec("ALTER TABLE conversations ADD COLUMN card_json TEXT NOT NULL DEFAULT '{}'");
  if (!convCols.has("card_version")) db.exec("ALTER TABLE conversations ADD COLUMN card_version INTEGER NOT NULL DEFAULT 0");
  if (!convCols.has("submit_message")) db.exec("ALTER TABLE conversations ADD COLUMN submit_message TEXT");
  // Новые пункты шаблона → новые колонки (старые данные не трогаем).
  const existing = new Set(
    (db.prepare("PRAGMA table_info(ideas)").all() as { name: string }[]).map((c) => c.name),
  );
  for (const k of FIELD_KEYS) {
    if (!existing.has(k)) db.exec(`ALTER TABLE ideas ADD COLUMN ${k} TEXT`);
  }
  return db;
}

/** Фото пока не храним: вырезаем base64 из сообщений, оставляем имя файла. */
function stripFileData(messages: UIMessage[]): UIMessage[] {
  return messages.map((m) => ({
    ...m,
    parts: m.parts.map((p) =>
      p.type === "file" ? { ...p, url: "" } : p,
    ) as UIMessage["parts"],
  }));
}

export function saveConversation(id: string, messages: UIMessage[], ip?: string) {
  getDb()
    .prepare(
      `INSERT INTO conversations (id, ip, message_count, messages_json) VALUES (?, ?, ?, ?)
       ON CONFLICT(id) DO UPDATE SET
         messages_json = excluded.messages_json,
         message_count = excluded.message_count,
         updated_at = datetime('now', 'localtime'),
         ip = COALESCE(conversations.ip, excluded.ip)`,
    )
    .run(id, ip ?? null, messages.length, JSON.stringify(stripFileData(messages)));
}

export type ConversationState = {
  messages: UIMessage[];
  card: Card;
  cardVersion: number;
  ideaId: number | null;
  submitMessage: string | null;
};

export function getConversation(id: string): ConversationState | undefined {
  const row = getDb()
    .prepare(
      "SELECT messages_json, card_json, card_version, idea_id, submit_message FROM conversations WHERE id = ?",
    )
    .get(id) as
    | { messages_json: string; card_json: string; card_version: number; idea_id: number | null; submit_message: string | null }
    | undefined;
  if (!row) return undefined;
  return {
    messages: JSON.parse(row.messages_json) as UIMessage[],
    card: JSON.parse(row.card_json) as Card,
    cardVersion: row.card_version,
    ideaId: row.idea_id,
    submitMessage: row.submit_message,
  };
}

export function saveCard(id: string, card: Card, version: number) {
  getDb()
    .prepare(
      `INSERT INTO conversations (id, card_json, card_version) VALUES (?, ?, ?)
       ON CONFLICT(id) DO UPDATE SET card_json = excluded.card_json, card_version = excluded.card_version`,
    )
    .run(id, JSON.stringify(card), version);
}

/** Сохраняет идею. Если по этому диалогу идея уже отправлена — возвращает её номер, не создавая дубль. */
export function insertIdea(conversationId: string, card: Card, dialog: string) {
  const d = getDb();
  const prev = d
    .prepare("SELECT idea_id, submit_message FROM conversations WHERE id = ?")
    .get(conversationId) as { idea_id: number | null; submit_message: string | null } | undefined;
  if (prev?.idea_id) return { id: prev.idea_id, duplicate: true, submitMessage: prev.submit_message };

  const cols = ["conversation_id", "anonymous", ...FIELD_KEYS, "dialog_text"];
  const values = [
    conversationId,
    card.anonymous ? 1 : 0,
    ...FIELD_KEYS.map((k) => (card.anonymous && k.startsWith("author_") ? null : (card[k] ?? null))),
    dialog,
  ];
  const res = d
    .prepare(`INSERT INTO ideas (${cols.join(", ")}) VALUES (${cols.map(() => "?").join(", ")})`)
    .run(...values);
  return { id: Number(res.lastInsertRowid), duplicate: false, submitMessage: null };
}

export function markSubmitted(conversationId: string, ideaId: number, message: string) {
  getDb()
    .prepare("UPDATE conversations SET idea_id = ?, submit_message = ? WHERE id = ?")
    .run(ideaId, message, conversationId);
}

export function listIdeas() {
  return getDb().prepare("SELECT * FROM ideas ORDER BY id DESC").all() as unknown as IdeaRow[];
}

export function getIdea(id: number) {
  return getDb().prepare("SELECT * FROM ideas WHERE id = ?").get(id) as unknown as
    | (IdeaRow & { dialog_text: string })
    | undefined;
}
