// Идентификатор текущего диалога живёт в localStorage: по нему сервер хранит переписку,
// а страница восстанавливает её после перезагрузки.
const KEY = "idea-chat:conversation-id";

let memoryId: string | null = null;

export function getConversationId(): string {
  try {
    let id = localStorage.getItem(KEY);
    if (!id) {
      id = crypto.randomUUID();
      localStorage.setItem(KEY, id);
    }
    return id;
  } catch {
    // Приватный режим / запрет хранилища — живём без восстановления.
    memoryId ??= crypto.randomUUID();
    return memoryId;
  }
}

/** Новая идея: забываем текущий диалог и перезагружаем страницу. */
export function startNewConversation() {
  try {
    localStorage.removeItem(KEY);
  } catch {
    /* ignore */
  }
  window.location.reload();
}
