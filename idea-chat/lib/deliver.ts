import { mkdir, appendFile, access } from "node:fs/promises";
import path from "node:path";
import nodemailer from "nodemailer";
import { IDEA_FIELDS, type Card, type Idea } from "./idea-template";
import { COMPANY_NAME } from "./company";
import { insertIdea, markSubmitted } from "./db";

export type Photo = { filename: string; mediaType: string; data: Buffer };

const DATA_DIR = path.join(process.cwd(), "data");
const CSV_PATH = path.join(DATA_DIR, "ideas.csv");

const csvCell = (v: string) => `"${v.replace(/"/g, '""').replace(/\r?\n/g, " ")}"`;

async function appendCsv(id: number, createdAt: string, idea: Idea, anonymous: boolean) {
  const header = ["Номер", "Дата", ...IDEA_FIELDS.map((f) => f.label), "Анонимно"];
  const row = [String(id), createdAt, ...IDEA_FIELDS.map((f) => idea[f.key] ?? ""), anonymous ? "да" : "нет"];
  await mkdir(DATA_DIR, { recursive: true });
  let exists = true;
  try {
    await access(CSV_PATH);
  } catch {
    exists = false;
  }
  // BOM + «;» — чтобы Excel с русской локалью открыл файл без мастера импорта.
  const head = exists ? "" : "﻿" + header.map(csvCell).join(";") + "\r\n";
  await appendFile(CSV_PATH, head + row.map(csvCell).join(";") + "\r\n", "utf8");
}

const escapeHtml = (s: string) =>
  s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

async function sendEmail(id: number, idea: Idea, anonymous: boolean, photos: Photo[], dialog: string) {
  const host = process.env.SMTP_HOST;
  const to = process.env.IDEAS_EMAIL_TO;
  if (!host || !to) return false;

  const transport = nodemailer.createTransport({
    host,
    port: Number(process.env.SMTP_PORT || 465),
    secure: (process.env.SMTP_SECURE ?? "true") !== "false",
    auth: process.env.SMTP_USER
      ? { user: process.env.SMTP_USER, pass: process.env.SMTP_PASS }
      : undefined,
  });

  const rows = IDEA_FIELDS.filter((f) => idea[f.key])
    .map(
      (f) =>
        `<tr><td style="padding:6px 12px;vertical-align:top;color:#555;white-space:nowrap"><b>${f.label}</b></td>` +
        `<td style="padding:6px 12px">${escapeHtml(idea[f.key]).replace(/\n/g, "<br>")}</td></tr>`,
    )
    .join("");

  await transport.sendMail({
    from: process.env.SMTP_FROM || process.env.SMTP_USER,
    to,
    subject: `Идея № ${id}: ${idea.title}`,
    html: `<p>Новая идея в «Банк идей» ${COMPANY_NAME}. Номер: <b>${id}</b>${anonymous ? " (анонимно)" : ""}</p><table style="border-collapse:collapse;font-family:sans-serif;font-size:14px">${rows}</table>`,
    text: IDEA_FIELDS.filter((f) => idea[f.key])
      .map((f) => `${f.label}: ${idea[f.key]}`)
      .join("\n"),
    attachments: [
      ...photos.map((p) => ({ filename: p.filename, content: p.data, contentType: p.mediaType })),
      { filename: `dialog-${id}.txt`, content: dialog, contentType: "text/plain; charset=utf-8" },
    ],
  });
  return true;
}

/** Текст после отправки — формирует код, а не модель, поэтому номер и контакт всегда настоящие. */
function submitMessage(id: number, card: Card) {
  if (card.anonymous) {
    return `Спасибо! Идея № ${id} отправлена анонимно. Её рассмотрят и проработают, но ответа по анонимным идеям не будет.`;
  }
  const name = card.author_name?.trim().split(/\s+/)[0];
  const thanks = name ? `Спасибо, ${name}!` : "Спасибо!";
  return card.author_contact?.trim()
    ? `${thanks} Идея № ${id} отправлена на рассмотрение. Ответ придёт вам: ${card.author_contact.trim()}.`
    : `${thanks} Идея № ${id} отправлена на рассмотрение. Контакт вы не оставили, поэтому ответ может до вас не дойти.`;
}

/**
 * Сохраняет идею в SQLite (основное хранилище), дублирует строкой в data/ideas.csv
 * и, если настроен SMTP, шлёт письмо. Повторная отправка из того же диалога не создаёт дубль.
 */
export async function deliverIdea(conversationId: string, card: Card, dialog: string) {
  // insertIdea и markSubmitted синхронны и идут подряд без await — двойной клик не создаст дубль.
  const res = insertIdea(conversationId, card, dialog);
  if (res.duplicate) return { id: res.id, message: res.submitMessage ?? submitMessage(res.id, card) };
  const message = submitMessage(res.id, card);
  markSubmitted(conversationId, res.id, message);

  const idea = (card.anonymous
    ? { ...card, author_name: "", author_department: "", author_contact: "" }
    : card) as Idea;
  const createdAt = new Date().toLocaleString("ru-RU");
  try {
    await appendCsv(res.id, createdAt, idea, !!card.anonymous);
  } catch (e) {
    console.error("Не удалось дописать идею в CSV", res.id, e);
  }
  try {
    await sendEmail(res.id, idea, !!card.anonymous, [], dialog);
  } catch (e) {
    console.error("Не удалось отправить письмо с идеей", res.id, e);
  }
  return { id: res.id, message };
}
