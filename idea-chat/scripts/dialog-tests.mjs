// Сценарии диалога с ботом. Запуск: node scripts/dialog-tests.mjs  (сервер на :3000 должен работать)
// Каждый сценарий — один прогон (это реальные запросы к Alice AI, они стоят денег).
// Ответы пользователя заранее заданы; бот может спрашивать в другом порядке — проверяем поведение, а не текст.

const BASE = process.env.BASE_URL || "http://localhost:3000";

const SCENARIOS = [
  {
    name: "Стульчик (реальный голосовой диалог пользователя)",
    expect: { author: "named" },
    replies: [
      "🎤 Привет у меня есть такая идея не знаю как правильно написать нам иногда просто неудобно стоять весь день на машине хочу стульчик",
      "🎤 Для дорожно разметочной для шмеля 12 с с какой стороны не знаю с той с которой удобно",
      "🎤 Да блин просто остановимся садимся передыхаем а так могли бы сразу отдыхать во время работы складной чтобы убирался",
      "🎤 Что за глупый вопрос сколько можно занять реализации не знаю мне кажется за 2 недели можно сделать а так это снизит усталость сотрудников и в целом может быть повысить продуктивность",
      "🎤 Не знаю оценить не могу наверное инженеры и сборщики должны этим заняться",
      "🎤 касается операторов машин которые на линии работают",
      "Меня зовут Олег Сидоров, оператор разметочной машины, телефон 8029 555 44 33",
      "Да",
    ],
  },
  {
    name: "Анонимная подача",
    expect: { author: "anonymous" },
    replies: [
      "Предлагаю поставить на складе готовой продукции термометр и записывать температуру — зимой краска АК-11 перемерзает и бывают возвраты.",
      "Кладовщик раз в день пишет в журнал. Касается склада. Термометр около 100 рублей, поставить за неделю. Польза — меньше возвратов.",
      "Не хочу называться, давайте анонимно",
      "Анонимно, да",
      "Больше добавить нечего",
    ],
  },
  {
    name: "Назвался, но контакт не даёт",
    expect: { author: "named", noContact: true },
    replies: [
      "Хочу, чтобы в столовой сделали предзаказ обедов через телеграм, а то очередь по 20 минут и обед пролетает.",
      "Касается всех сотрудников офиса и цеха. Нужен программист на месяц, денег немного. Польза — экономия времени, люди не опаздывают с обеда.",
      "Наверное месяц-два на всё. Начать можно с бумажного списка заказов утром.",
      "Анна Ковалёва, бухгалтерия",
      "Нет, телефон давать не буду",
      "Не хочу оставлять контакт",
    ],
  },
  {
    name: "Посторонние вопросы посреди диалога",
    expect: { author: "named" },
    replies: [
      "Какая завтра погода в Бресте?",
      "Напиши мне код на питоне для сортировки",
      "Ладно. Идея: в лаборатории сделать электронный журнал испытаний вместо бумажного, а то протоколы теряются.",
      "Касается испытательной лаборатории. Нужен планшет и программа, может тысяча рублей. Польза — протоколы не теряются, быстрее поиск. Срок месяц.",
      "Пётр Иванов, лаборатория, инженер, petr@stim.by",
      "Всё",
    ],
  },
  {
    name: "Всё одним сообщением",
    expect: { author: "named" },
    replies: [
      "Идея: на отгрузке проверять комплектацию машин по чек-листу. Сейчас мастер смотрит на глаз и раз в месяц забывают форсунки или фильтры, сервис потом досылает. Сделать чек-лист на каждую модель, мастер отмечает и фотографирует комплект. Касается цеха сборки и отгрузки. Нужен технолог на пару дней, денег почти не надо. Запустить за месяц. Польза — меньше досылок и претензий клиентов. Я Иван Петров, цех сборки, слесарь, телефон 8029 123 45 67.",
      "Да, всё так",
    ],
  },
  {
    name: "Правка после карточки",
    expect: { author: "named", edit: true },
    replies: [
      "Предлагаю на складе сырья поставить стеллажи с подписями под смолы, сейчас ищем по полчаса. Касается склада сырья. Нужны стеллажи примерно на 2000 рублей, срок две недели. Польза — экономия времени кладовщиков. Мария Белова, склад сырья, кладовщик, 8033 777 88 99",
    ],
    afterCard: "Поправьте срок: не две недели, а два месяца",
  },
  {
    name: "Короткие ответы",
    expect: { author: "anonymous" },
    replies: [
      "хочу чтобы в цеху было светлее",
      "темно",
      "лампы поменять",
      "не знаю",
      "цех сборки",
      "не знаю сколько",
      "ну месяц",
      "анонимно",
      "нет",
    ],
  },
];

const BAD = /(Пользователь|Ассистент|Assistant|User)\s*:|```|\{\s*"|submit_idea|номер\w* заявк|\[номер\]/i;

function parseStream(text) {
  const events = text
    .split("\n")
    .filter((l) => l.startsWith("data: {"))
    .map((l) => JSON.parse(l.slice(6)));
  return {
    text: events.filter((e) => e.type === "text-delta").map((e) => e.delta).join(""),
    card: events.find((e) => e.type === "tool-input-available" && e.toolName === "show_card")?.input,
  };
}

async function chat(conversationId, messages) {
  const res = await fetch(`${BASE}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ conversationId, messages }),
  });
  if (!res.ok) return { status: res.status, error: await res.text() };
  return { status: 200, ...parseStream(await res.text()) };
}

async function restore(conversationId) {
  return (await (await fetch(`${BASE}/api/conversation?id=${conversationId}`)).json()).messages;
}

async function submit(conversationId, version) {
  const res = await fetch(`${BASE}/api/submit`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ conversationId, version }),
  });
  return { status: res.status, ...(await res.json()) };
}

async function runScenario(sc, idx) {
  const conv = `test-${Date.now()}-${idx}`;
  const problems = [];
  const log = [];
  let messages = [];
  let card = null;
  let turns = 0;

  const say = async (text) => {
    messages.push({ id: `u${messages.length}`, role: "user", parts: [{ type: "text", text }] });
    const r = await chat(conv, messages);
    turns++;
    if (r.status !== 200) {
      problems.push(`HTTP ${r.status}: ${r.error}`);
      return r;
    }
    log.push(`👤 ${text.slice(0, 90)}\n🤖 ${r.text}${r.card ? `  [КАРТОЧКА v${r.card.version}]` : ""}`);
    if (BAD.test(r.text)) problems.push(`запрещённое в реплике: «${r.text.slice(0, 120)}»`);
    if (r.text.length > 500) problems.push(`слишком длинная реплика (${r.text.length})`);
    messages = await restore(conv);
    return r;
  };

  for (const reply of sc.replies) {
    const r = await say(reply);
    if (r.card) {
      card = r.card;
      break;
    }
  }

  if (!card) {
    problems.push("карточка так и не показана");
  } else {
    const c = card.card;
    const authorOk =
      sc.expect.author === "anonymous" ? c.anonymous === true : !c.anonymous && !!c.author_name;
    if (!authorOk) problems.push(`автор: ожидалось ${sc.expect.author}, получено ${JSON.stringify({ anonymous: c.anonymous, name: c.author_name, contact: c.author_contact })}`);

    if (sc.afterCard) {
      const r = await say(sc.afterCard);
      if (!r.card) problems.push("после правки карточка не обновилась");
      else {
        const stale = await submit(conv, card.version);
        if (stale.status !== 409) problems.push(`отправка старой версии не отклонена (${stale.status})`);
        card = r.card;
        if (!/месяц/i.test(card.card.timeline ?? "")) problems.push(`срок не исправлен: ${card.card.timeline}`);
      }
    }

    const s = await submit(conv, card.version);
    if (s.status !== 200 || !s.id) problems.push(`отправка не удалась: ${JSON.stringify(s)}`);
    else {
      log.push(`✅ ${s.message}`);
      if (!s.message.includes(`№ ${s.id}`)) problems.push("в сообщении нет номера");
      if (sc.expect.author === "anonymous" && !/анонимно/.test(s.message)) problems.push("нет пометки об анонимности");
      if (sc.expect.noContact && !/не оставили/.test(s.message)) problems.push("не сказано, что контакта нет");
      if (card.card.author_contact && !s.message.includes(card.card.author_contact)) problems.push("контакт не назван в итоговом сообщении");
      const again = await submit(conv, card.version);
      if (again.id !== s.id) problems.push("повторная отправка дала другой номер");
      const after = await chat(conv, [...messages, { id: "x", role: "user", parts: [{ type: "text", text: "ещё вопрос" }] }]);
      if (after.status !== 409) problems.push(`после отправки чат не закрыт (${after.status})`);
    }
  }

  return { name: sc.name, turns, problems, log };
}

const only = process.argv[2] ? Number(process.argv[2]) : null;
let failed = 0;
for (const [i, sc] of SCENARIOS.entries()) {
  if (only !== null && only !== i + 1) continue;
  const r = await runScenario(sc, i);
  const ok = r.problems.length === 0;
  if (!ok) failed++;
  console.log(`\n=== ${i + 1}. ${r.name} — ${ok ? "OK" : "ПРОБЛЕМЫ"} (ходов: ${r.turns})`);
  console.log(r.log.join("\n"));
  for (const p of r.problems) console.log(`  ❌ ${p}`);
}
console.log(`\nИтого: ${only ? 1 : SCENARIOS.length} сценариев, с проблемами: ${failed}`);
