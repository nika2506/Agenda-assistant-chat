const form = document.querySelector('#question-form');
const input = document.querySelector('#question');
const chat = document.querySelector('#chat');
const emptyState = document.querySelector('#empty-state');
const status = document.querySelector('#status');
const sendButton = document.querySelector('#send');
const template = document.querySelector('#message-template');

function addMessage(role, text, sources = []) {
  emptyState.hidden = true;
  const message = template.content.firstElementChild.cloneNode(true);
  message.classList.add(role);
  message.querySelector('.message-role').textContent = role === 'user' ? 'You' : 'Agenda assistant';
  message.querySelector('.message-content').textContent = text;
  if (sources.length) {
    const details = document.createElement('details');
    details.className = 'sources';
    details.innerHTML = `<summary>Grounding sources (${sources.length})</summary>`;
    sources.forEach((source) => {
      const item = document.createElement('pre');
      item.textContent = `[${source.id}] ${source.label}\n${source.content}`;
      details.append(item);
    });
    message.append(details);
  }
  chat.append(message);
  message.scrollIntoView({ behavior: 'smooth', block: 'end' });
}

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  const question = input.value.trim();
  if (!question) return;

  addMessage('user', question);
  input.value = '';
  input.disabled = true;
  sendButton.disabled = true;
  status.textContent = 'Searching the agenda and drafting a grounded answer...';

  try {
    const response = await fetch('/api/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question }),
    });
    const body = await response.json();
    if (!response.ok) throw new Error(body.detail || 'The backend could not answer right now.');
    addMessage('assistant', body.answer, body.sources);
    status.textContent = '';
  } catch (error) {
    addMessage('error', `I couldn't complete that request. ${error.message}`);
    status.textContent = 'You can try again once the backend or local model is available.';
  } finally {
    input.disabled = false;
    sendButton.disabled = false;
    input.focus();
  }
});
