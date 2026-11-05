const messagesArea = document.getElementById('messagesArea');
const queryInput = document.getElementById('queryInput');
const sendBtn = document.getElementById('sendBtn');
const typingIndicator = document.getElementById('typingIndicator');

function scrollToBottom() {
  messagesArea.scrollTop = messagesArea.scrollHeight;
}

function handleChipClick(questionText) {
  queryInput.value = questionText;
  handleFormSubmit(new Event('submit'));
}

async function handleFormSubmit(e) {
  if (e) e.preventDefault();
  const query = queryInput.value.trim();
  if (!query) return;

  // 1. Render User Message
  appendUserMessage(query);
  queryInput.value = '';
  queryInput.disabled = true;
  sendBtn.disabled = true;

  // 2. Show typing indicator
  typingIndicator.style.display = 'flex';
  scrollToBottom();

  try {
    const response = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: query })
    });

    const data = await response.json();
    typingIndicator.style.display = 'none';

    if (response.ok && data.answer) {
      appendAssistantMessage(data.answer, data.sources || []);
    } else {
      appendAssistantMessage(data.error || "Sorry, an error occurred while processing your query.", []);
    }
  } catch (err) {
    typingIndicator.style.display = 'none';
    appendAssistantMessage("Network error: Could not reach the Clarus server.", []);
    console.error(err);
  } finally {
    queryInput.disabled = false;
    sendBtn.disabled = false;
    queryInput.focus();
    scrollToBottom();
  }
}

function appendUserMessage(text) {
  const msgDiv = document.createElement('div');
  msgDiv.className = 'message user-message';
  msgDiv.innerHTML = `
    <div class="avatar-badge">You</div>
    <div class="message-content">
      <div class="message-bubble"><p>${escapeHtml(text)}</p></div>
    </div>
  `;
  messagesArea.appendChild(msgDiv);
  scrollToBottom();
}

function appendAssistantMessage(rawText, sources) {
  const msgDiv = document.createElement('div');
  msgDiv.className = 'message assistant-message';

  let sourcesHtml = '';
  if (sources && sources.length > 0) {
    sourcesHtml = `
      <div class="sources-card">
        <div class="sources-card-title">Verified Sources:</div>
        ${sources.map(s => `
          <div class="source-item">
            <span>📄</span>
            <a href="${escapeHtml(s.url)}" target="_blank" rel="noopener noreferrer">${escapeHtml(s.title || s.file)}</a>
          </div>
        `).join('')}
      </div>
    `;
  }

  msgDiv.innerHTML = `
    <div class="avatar-badge">AI</div>
    <div class="message-content">
      <div class="message-bubble">
        ${formatMarkdown(rawText)}
      </div>
      ${sourcesHtml}
    </div>
  `;
  messagesArea.appendChild(msgDiv);
  scrollToBottom();
}

function escapeHtml(str) {
  return String(str).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

function formatMarkdown(text) {
  if (!text) return '';
  let html = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>').replace(/\*(.*?)\*/g, '<em>$1</em>');
  const lines = html.split('\n');
  let inList = false;
  let formatted = [];

  for (let line of lines) {
    let trimmed = line.trim();
    if (trimmed.startsWith('- ') || trimmed.startsWith('* ')) {
      if (!inList) { formatted.push('<ul>'); inList = true; }
      formatted.push(`<li>${trimmed.substring(2)}</li>`);
    } else {
      if (inList) { formatted.push('</ul>'); inList = false; }
      if (trimmed.length > 0) formatted.push(`<p>${trimmed}</p>`);
    }
  }
  if (inList) formatted.push('</ul>');
  return formatted.join('');
}
