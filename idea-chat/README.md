# Банк идей СТиМ — чат вместо анкеты

Одностраничник на [assistant-ui](https://github.com/assistant-ui/assistant-ui) (Next.js).
Сотрудник рассказывает идею текстом, голосовым или с фото, бот (Yandex Alice AI) задаёт уточняющие
вопросы по шаблону, показывает итоговую карточку и после подтверждения отправляет её.

## Запуск
```bash
npm install
cp .env.example .env.local   # заполнить YANDEX_API_KEY, YANDEX_FOLDER_ID
npm run dev                  # http://localhost:3000
```

## Где что менять
- `lib/idea-template.ts` — пункты шаблона идеи (порядок, обязательность, подсказки).
- `lib/company.ts` — контекст компании для бота.
- `lib/prompt.ts` — системный промпт: стиль, отказ от посторонних тем, завершение.
- `lib/db.ts` — SQLite `data/ideas.db`: таблица `ideas` (номер, пункты шаблона, анонимность, статус,
  текст диалога) и `conversations` (вся переписка JSON-ом). Открывается, например, DB Browser for SQLite.
- `lib/deliver.ts` — куда уходит готовая идея: SQLite + дубль строкой в `data/ideas.csv` + письмо,
  если задан SMTP. Сюда же потом — Google Таблица и задача в Bitrix24.
- `/admin` — просмотр идей (пароль `ADMIN_PASSWORD` в `.env.local`, логин любой).
- `lib/rate-limit.ts` — лимит: 20 запросов на диалог и 200 на IP за 10 минут (настраивается в env).
- `app/api/chat/route.ts` — вызов модели, инструмент `submit_idea`, обработка фото.
- `app/api/transcribe/route.ts` + `components/voice-recorder.tsx` — голосовые через SpeechKit.

## Особенности Yandex
- Alice AI не принимает картинки: фото заменяются пометкой и прикладываются к заявке.
  Если в каталоге доступна vision-модель — укажите её в `YANDEX_VISION_MODEL`, бот получит описание фото.
- В стриминге с tool calls Яндекс не присылает `finish_reason` — это чинит `lib/yandex-fetch.ts`.
