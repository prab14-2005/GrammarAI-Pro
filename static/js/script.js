/**
 * Grammar Error Detector & Corrector - Frontend Script
 * Micro-project: Natural Language Processing
 */

// Application State
let currentErrors = [];
let currentAnalysis = null;
let currentCorrectedText = "";
let historyRecords = [];

// DOM Ready initialization
document.addEventListener('DOMContentLoaded', () => {
  initTheme();
  initNavigation();
  initInputListeners();
  loadDashboardData();
  loadHistoryData();
});

// -------------------------------------------------------------
// 1. Theme Management (Light / Dark Mode)
// -------------------------------------------------------------
function initTheme() {
  const savedTheme = localStorage.getItem('nlp-grammar-theme') || 'dark';
  document.documentElement.setAttribute('data-theme', savedTheme);

  const themeBtn = document.getElementById('theme-toggle-btn');
  if (themeBtn) {
    themeBtn.addEventListener('click', () => {
      const current = document.documentElement.getAttribute('data-theme');
      const nextTheme = current === 'dark' ? 'light' : 'dark';
      document.documentElement.setAttribute('data-theme', nextTheme);
      localStorage.setItem('nlp-grammar-theme', nextTheme);
      showToast(`Switched to ${nextTheme} theme`, 'info');
    });
  }
}

// -------------------------------------------------------------
// 2. Tab Navigation
// -------------------------------------------------------------
function initNavigation() {
  const tabButtons = document.querySelectorAll('.nav-tab');
  tabButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      const targetTab = btn.getAttribute('data-tab');
      switchTab(targetTab);
    });
  });

  // Handle URL hash if present
  const hash = window.location.hash.replace('#', '');
  if (hash && document.getElementById(`section-${hash}`)) {
    switchTab(hash);
  }
}

function switchTab(tabId) {
  // Update nav buttons
  document.querySelectorAll('.nav-tab').forEach(btn => {
    const isTarget = btn.getAttribute('data-tab') === tabId;
    btn.classList.toggle('active', isTarget);
    btn.setAttribute('aria-selected', isTarget);
  });

  // Update tab sections
  document.querySelectorAll('.tab-content').forEach(sec => {
    sec.classList.remove('active');
  });

  const targetSection = document.getElementById(`section-${tabId}`);
  if (targetSection) {
    targetSection.classList.add('active');
    window.location.hash = tabId;
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }

  // Refresh tab-specific data
  if (tabId === 'dashboard') {
    loadDashboardData();
  } else if (tabId === 'history') {
    loadHistoryData();
  }
}

// -------------------------------------------------------------
// 3. Input Textarea Counters & Quick Presets
// -------------------------------------------------------------
function initInputListeners() {
  const textarea = document.getElementById('input-text');
  if (!textarea) return;

  textarea.addEventListener('input', () => {
    updateCounters(textarea.value);
  });

  // Ctrl+Enter or Cmd+Enter to run check
  textarea.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
      e.preventDefault();
      handleGrammarCheck();
    }
  });
}

function updateCounters(text) {
  const trimmed = text.trim();
  const wordCount = trimmed ? trimmed.split(/\s+/).length : 0;
  const charCount = text.length;

  const wordEl = document.getElementById('input-word-count');
  const charEl = document.getElementById('input-char-count');
  if (wordEl) wordEl.textContent = `${wordCount} word${wordCount === 1 ? '' : 's'}`;
  if (charEl) charEl.textContent = `${charCount} character${charCount === 1 ? '' : 's'}`;
}

function clearInput() {
  const textarea = document.getElementById('input-text');
  if (textarea) {
    textarea.value = '';
    updateCounters('');
    textarea.focus();
  }
  const results = document.getElementById('results-wrapper');
  if (results) results.classList.add('hidden');
  closeInspector();
}

function onSelectSample(val) {
  if (!val) return;
  runSample(val);
}

function runSample(sentence) {
  const textarea = document.getElementById('input-text');
  if (textarea) {
    textarea.value = sentence;
    updateCounters(sentence);
  }
  switchTab('checker');
  handleGrammarCheck();
}

// -------------------------------------------------------------
// 4. Grammar Checking Logic
// -------------------------------------------------------------
async function handleGrammarCheck() {
  const textarea = document.getElementById('input-text');
  const text = textarea ? textarea.value.trim() : '';

  if (!text) {
    showToast('Please enter some text to check grammar.', 'error');
    if (textarea) textarea.focus();
    return;
  }

  const checkBtn = document.getElementById('btn-check-grammar');
  if (checkBtn) checkBtn.classList.add('loading');

  try {
    const response = await fetch('/api/check', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: text })
    });

    const data = await response.json();

    if (!response.ok || !data.success) {
      throw new Error(data.error || 'Server returned an error.');
    }

    // Save state
    currentErrors = data.errors || [];
    currentAnalysis = data.nlp_analysis || null;
    currentCorrectedText = data.corrected_text || '';

    // Render results
    renderResults(data);
    showToast(`Checked successfully: ${data.error_count} error${data.error_count === 1 ? '' : 's'} detected.`, 'success');

    // Refresh telemetry in background
    loadDashboardData();

  } catch (err) {
    showToast(err.message || 'Failed to complete grammar check.', 'error');
  } finally {
    if (checkBtn) checkBtn.classList.remove('loading');
  }
}

async function handleAICorrection() {
  const textarea = document.getElementById('input-text');
  const text = textarea ? textarea.value.trim() : '';
  if (!text) { showToast('Please enter text for AI correction.', 'error'); return; }
  const button = document.getElementById('btn-ai-correct');
  const overlay = document.getElementById('ai-loading-overlay');
  if (button) button.classList.add('loading');
  if (button) button.disabled = true;
  if (overlay) overlay.classList.remove('hidden');
  try {
    const response = await fetch('/api/ai-correct', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({text})});
    const data = await response.json();
    if (!response.ok || !data.success) throw new Error(data.error || 'AI correction failed.');
    currentCorrectedText = data.corrected_text;
    const output = document.getElementById('ai-corrected-text-display');
    const details = document.getElementById('ai-correction-result');
    const model = document.getElementById('ai-model-name');
    if (output) output.textContent = data.corrected_text;
    if (model) model.textContent = `Model: ${data.model}`;
    if (details) details.classList.remove('hidden');
    populateNLPAnalysis(data.nlp_analysis);
    showToast('AI correction completed. Please review the meaning.', 'success');
    loadDashboardData();
  } catch (err) { showToast(err.message || 'AI correction failed.', 'error'); }
  finally {
    if (button) { button.classList.remove('loading'); button.disabled = false; }
    if (overlay) overlay.classList.add('hidden');
  }
}

// -------------------------------------------------------------
// 5. Render Results & Highlights
// -------------------------------------------------------------
function renderResults(data) {
  const wrapper = document.getElementById('results-wrapper');
  if (wrapper) wrapper.classList.remove('hidden');

  // 1. Update Statistical Badges & Circular Progress Gauge
  const stats = data.statistics || {};
  setText('stat-words', stats.total_words || 0);
  setText('stat-sentences', stats.total_sentences || 0);
  setText('stat-errors', stats.errors_detected || 0);
  setText('stat-corrections', stats.corrections_made || 0);
  setText('stat-accuracy', `${data.accuracy_percentage || 100}%`);
  updateCircularScore(data.accuracy_percentage || 100);

  // 2. Render Interactive Highlighted Display
  const highlightBox = document.getElementById('highlighted-display');
  if (highlightBox) {
    highlightBox.innerHTML = data.highlighted_html || escapeHtml(data.original_text);

    // Attach click listeners to error spans
    const errorSpans = highlightBox.querySelectorAll('.error-highlight');
    errorSpans.forEach(span => {
      span.addEventListener('click', () => {
        // Highlight active span
        errorSpans.forEach(s => s.classList.remove('active-error'));
        span.classList.add('active-error');

        const errId = parseInt(span.getAttribute('data-error-id'), 10);
        const errObj = currentErrors.find(e => e.id === errId);
        if (errObj) {
          openInspector(errObj);
        }
      });
    });
  }

  // 3. Render Error Breakdown Table
  renderErrorsTable(data.errors);

  // 4. Render Corrected Text
  const correctedBox = document.getElementById('corrected-text-display');
  if (correctedBox) {
    correctedBox.textContent = data.corrected_text;
  }

  // 5. Populate NLP Analysis Tab
  populateNLPAnalysis(data.nlp_analysis);
}

function updateCircularScore(scoreVal) {
  const circleBar = document.getElementById('score-circle-bar');
  if (!circleBar) return;
  const num = Math.min(100, Math.max(0, parseFloat(scoreVal) || 0));
  const circumference = 301.59;
  const offset = circumference - (num / 100) * circumference;
  circleBar.style.strokeDashoffset = offset;

  // Circular progress color shift: Blue, Cyan, Green depending on score
  if (num >= 90) {
    circleBar.style.stroke = '#22C55E'; // Green
  } else if (num >= 75) {
    circleBar.style.stroke = '#06B6D4'; // Cyan
  } else if (num >= 50) {
    circleBar.style.stroke = '#2563EB'; // Blue
  } else {
    circleBar.style.stroke = '#EF4444'; // Red
  }
}

function renderErrorsTable(errors) {
  const tbody = document.getElementById('errors-table-body');
  const countBadge = document.getElementById('error-count-badge');
  if (countBadge) {
    countBadge.textContent = `${errors.length} Error${errors.length === 1 ? '' : 's'}`;
  }

  if (!tbody) return;
  tbody.innerHTML = '';

  if (!errors || errors.length === 0) {
    tbody.innerHTML = `
      <tr>
        <td colspan="6" class="text-center" style="color: var(--success); padding: 1.5rem;">
          ✔ No grammatical errors detected! Great job.
        </td>
      </tr>
    `;
    return;
  }

  errors.forEach((err, idx) => {
    const tr = document.createElement('tr');
    const cat = (err.category || 'grammar').toLowerCase();
    tr.className = `err-row-${cat}`;
    const suggestions = err.suggestions || [];
    const bestSuggestion = suggestions.length > 0 ? suggestions[0] : 'None';

    tr.innerHTML = `
      <td><strong>${idx + 1}</strong></td>
      <td><span class="badge error-${cat}">${escapeHtml(err.error_text || '—')}</span></td>
      <td><span class="badge badge-subtle">${cat.toUpperCase()}</span></td>
      <td><strong style="color: var(--primary);">${escapeHtml(bestSuggestion)}</strong></td>
      <td style="max-width: 320px; font-size: 0.85rem; color: var(--text-secondary);">${escapeHtml(err.explanation || 'Language error')}</td>
      <td>
        <button class="btn btn-secondary btn-sm" onclick="applySingleFix(${err.id})">
          Apply Fix
        </button>
      </td>
    `;
    tbody.appendChild(tr);
  });
}

// -------------------------------------------------------------
// 6. Interactive Error Inspector
// -------------------------------------------------------------
function openInspector(err) {
  const drawer = document.getElementById('error-inspector');
  if (!drawer) return;

  drawer.classList.remove('hidden');

  const wordEl = document.getElementById('inspector-error-word');
  const badgeEl = document.getElementById('inspector-badge');
  const explanationEl = document.getElementById('inspector-explanation');
  const suggestionsBox = document.getElementById('inspector-suggestions');

  const cat = (err.category || 'grammar').toLowerCase();
  if (cat === 'grammar') {
    drawer.style.borderLeftColor = 'var(--err-grammar)';
  } else if (cat === 'spelling') {
    drawer.style.borderLeftColor = 'var(--err-spelling)';
  } else if (cat === 'punctuation') {
    drawer.style.borderLeftColor = 'var(--err-punctuation)';
  } else {
    drawer.style.borderLeftColor = 'var(--err-style)';
  }

  if (wordEl) wordEl.textContent = `"${err.error_text}"`;
  if (badgeEl) {
    badgeEl.className = `badge error-${cat}`;
    badgeEl.textContent = `${cat.toUpperCase()} ERROR`;
  }
  if (explanationEl) {
    explanationEl.textContent = err.explanation || 'Grammatical rule suggestion.';
  }

  if (suggestionsBox) {
    suggestionsBox.innerHTML = '';
    const suggestions = err.suggestions || [];

    if (suggestions.length === 0) {
      suggestionsBox.innerHTML = '<span style="color: var(--text-muted);">No automatic suggestion available.</span>';
    } else {
      suggestions.forEach(sug => {
        const btn = document.createElement('button');
        btn.className = 'btn-suggestion';
        btn.innerHTML = `<span>✔ Replace with <strong>${escapeHtml(sug)}</strong></span>`;
        btn.onclick = () => {
          applySuggestionToInput(err, sug);
        };
        suggestionsBox.appendChild(btn);
      });
    }
  }

  drawer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function closeInspector() {
  const drawer = document.getElementById('error-inspector');
  if (drawer) drawer.classList.add('hidden');
  document.querySelectorAll('.error-highlight').forEach(s => s.classList.remove('active-error'));
}

function applySingleFix(errId) {
  const err = currentErrors.find(e => e.id === errId);
  if (!err || !err.suggestions || err.suggestions.length === 0) return;
  applySuggestionToInput(err, err.suggestions[0]);
}

function applySuggestionToInput(err, replacement) {
  const textarea = document.getElementById('input-text');
  if (!textarea) return;

  const original = textarea.value;
  const start = err.offset;
  const end = err.offset + err.length;

  if (start <= original.length) {
    const updated = original.slice(0, start) + replacement + original.slice(end);
    textarea.value = updated;
    updateCounters(updated);
    showToast(`Applied fix: "${err.error_text}" &rarr; "${replacement}"`, 'success');
    closeInspector();
    // Re-check automatically
    handleGrammarCheck();
  }
}

// -------------------------------------------------------------
// 7. Corrected Text Utilities
// -------------------------------------------------------------
function copyCorrectedText() {
  if (!currentCorrectedText) return;
  navigator.clipboard.writeText(currentCorrectedText).then(() => {
    const btnText = document.getElementById('copy-btn-text');
    if (btnText) btnText.textContent = 'Copied!';
    showToast('Corrected text copied to clipboard!', 'success');
    setTimeout(() => {
      if (btnText) btnText.textContent = 'Copy Text';
    }, 2000);
  }).catch(() => {
    showToast('Failed to copy text.', 'error');
  });
}

function applyAndRecheck() {
  if (!currentCorrectedText) return;
  const textarea = document.getElementById('input-text');
  if (textarea) {
    textarea.value = currentCorrectedText;
    updateCounters(currentCorrectedText);
    handleGrammarCheck();
  }
}

// -------------------------------------------------------------
// 8. NLP Analysis Rendering
// -------------------------------------------------------------
function populateNLPAnalysis(analysis) {
  if (!analysis) return;

  setText('nlp-sentence-count', analysis.sentence_count || 0);
  setText('nlp-word-count', analysis.word_count || 0);
  setText('nlp-unique-words', analysis.unique_words || 0);
  setText('nlp-lexical-diversity', `${analysis.lexical_diversity || 0}%`);

  const tbody = document.getElementById('tokens-table-body');
  if (!tbody) return;
  tbody.innerHTML = '';

  const tokens = analysis.tokens || [];
  if (tokens.length === 0) {
    tbody.innerHTML = '<tr><td colspan="6" class="text-center empty-state">No tokens available.</td></tr>';
    return;
  }

  tokens.forEach(tok => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${tok.index}</td>
      <td><strong>${escapeHtml(tok.token)}</strong></td>
      <td><span class="badge ${tok.color_class || 'tag-other'}">${escapeHtml(tok.tag)}</span></td>
      <td>${escapeHtml(tok.category || 'Other')}</td>
      <td style="color: var(--text-secondary); font-size: 0.85rem;">${escapeHtml(tok.description || '')}</td>
      <td><code style="color: var(--accent);">${escapeHtml(tok.lemma || tok.token)}</code></td>
    `;
    tbody.appendChild(tr);
  });
}

// -------------------------------------------------------------
// 9. Dashboard Telemetry
// -------------------------------------------------------------
async function loadDashboardData() {
  try {
    const [statsRes, historyRes] = await Promise.all([
      fetch('/api/stats'),
      fetch('/api/history?limit=5')
    ]);

    if (statsRes.ok) {
      const statsData = await statsRes.json();
      if (statsData.success) {
        updateDashboardStats(statsData.statistics);
      }
    }

    if (historyRes.ok) {
      const histData = await historyRes.json();
      if (histData.success) {
        renderRecentChecks(histData.history);
      }
    }
  } catch (e) {
    console.error('Error loading dashboard telemetry:', e);
  }
}

function updateDashboardStats(stats) {
  if (!stats) return;

  setText('dash-total-checks', stats.total_checks || 0);
  setText('dash-total-errors', stats.total_errors || 0);
  setText('dash-total-corrections', stats.total_corrections || 0);
  setText('dash-avg-accuracy', `${stats.avg_accuracy || 100}%`);

  const cats = stats.category_counts || {};
  const grammarCount = cats.grammar || 0;
  const spellingCount = cats.spelling || 0;
  const punctuationCount = cats.punctuation || 0;
  const styleCount = (cats.style || 0) + (cats.other || 0);
  const totalErrors = Math.max(1, grammarCount + spellingCount + punctuationCount + styleCount);

  setText('count-grammar', grammarCount);
  setText('count-spelling', spellingCount);
  setText('count-punctuation', punctuationCount);
  setText('count-style', styleCount);

  setBarWidth('bar-grammar', (grammarCount / totalErrors) * 100);
  setBarWidth('bar-spelling', (spellingCount / totalErrors) * 100);
  setBarWidth('bar-punctuation', (punctuationCount / totalErrors) * 100);
  setBarWidth('bar-style', (styleCount / totalErrors) * 100);
}

function renderRecentChecks(recentList) {
  const tbody = document.getElementById('dashboard-recent-body');
  if (!tbody) return;
  tbody.innerHTML = '';

  if (!recentList || recentList.length === 0) {
    tbody.innerHTML = '<tr><td colspan="6" class="text-center empty-state">No checks recorded yet. Run a check to see data!</td></tr>';
    return;
  }

  recentList.forEach(item => {
    const tr = document.createElement('tr');
    const acc = item.nlp_summary && item.nlp_summary.accuracy !== undefined ? `${item.nlp_summary.accuracy}%` : '—';
    tr.innerHTML = `
      <td style="max-width: 240px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">${escapeHtml(item.original_text)}</td>
      <td style="max-width: 240px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: var(--success);">${escapeHtml(item.corrected_text)}</td>
      <td><span class="badge ${item.error_count > 0 ? 'badge-danger' : 'badge-success'}">${item.error_count}</span></td>
      <td><strong>${acc}</strong></td>
      <td style="font-size: 0.8rem; color: var(--text-muted);">${escapeHtml(item.created_at || '')}</td>
      <td>
        <button class="btn btn-secondary btn-sm" onclick="loadHistoryItemIntoChecker('${encodeURIComponent(item.original_text)}')">
          Load
        </button>
      </td>
    `;
    tbody.appendChild(tr);
  });
}

// -------------------------------------------------------------
// 10. History Storage & CRUD
// -------------------------------------------------------------
async function loadHistoryData() {
  try {
    const res = await fetch('/api/history?limit=100');
    if (!res.ok) throw new Error('Failed to fetch history');

    const data = await res.json();
    historyRecords = data.history || [];

    const badge = document.getElementById('history-total-badge');
    if (badge) badge.textContent = `${historyRecords.length} Record${historyRecords.length === 1 ? '' : 's'}`;

    renderHistoryTable(historyRecords);
  } catch (err) {
    console.error('History error:', err);
    const tbody = document.getElementById('history-table-body');
    if (tbody) {
      tbody.innerHTML = `<tr><td colspan="7" class="text-center" style="color: var(--danger);">Failed to load history: ${err.message}</td></tr>`;
    }
  }
}

function renderHistoryTable(records) {
  const tbody = document.getElementById('history-table-body');
  if (!tbody) return;
  tbody.innerHTML = '';

  if (!records || records.length === 0) {
    tbody.innerHTML = '<tr><td colspan="7" class="text-center empty-state">No grammar check history found.</td></tr>';
    return;
  }

  records.forEach(rec => {
    const tr = document.createElement('tr');
    const acc = rec.nlp_summary && rec.nlp_summary.accuracy !== undefined ? `${rec.nlp_summary.accuracy}%` : '100%';
    tr.innerHTML = `
      <td>#${rec.id}</td>
      <td style="max-width: 250px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;" title="${escapeHtml(rec.original_text)}">${escapeHtml(rec.original_text)}</td>
      <td style="max-width: 250px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: var(--success);" title="${escapeHtml(rec.corrected_text)}">${escapeHtml(rec.corrected_text)}</td>
      <td><span class="badge ${rec.error_count > 0 ? 'badge-danger' : 'badge-success'}">${rec.error_count}</span></td>
      <td><strong>${acc}</strong></td>
      <td style="font-size: 0.8rem; color: var(--text-muted);">${escapeHtml(rec.created_at || '')}</td>
      <td>
        <div style="display: flex; gap: 0.4rem;">
          <button class="btn btn-secondary btn-sm" onclick="showHistoryDetail(${rec.id})">Details</button>
          <button class="btn btn-danger btn-sm" onclick="deleteHistoryItem(${rec.id})" title="Delete check">&times;</button>
        </div>
      </td>
    `;
    tbody.appendChild(tr);
  });
}

function filterHistory(query) {
  const q = (query || '').toLowerCase().trim();
  if (!q) {
    renderHistoryTable(historyRecords);
    return;
  }
  const filtered = historyRecords.filter(r => 
    r.original_text.toLowerCase().includes(q) ||
    r.corrected_text.toLowerCase().includes(q)
  );
  renderHistoryTable(filtered);
}

async function deleteHistoryItem(id) {
  if (!confirm('Are you sure you want to delete this check from history?')) return;
  try {
    const res = await fetch(`/api/history/${id}`, { method: 'DELETE' });
    const data = await res.json();
    if (res.ok && data.success) {
      showToast('Record deleted.', 'info');
      loadHistoryData();
      loadDashboardData();
    } else {
      showToast(data.error || 'Failed to delete record.', 'error');
    }
  } catch (err) {
    showToast(err.message, 'error');
  }
}

async function confirmClearHistory() {
  if (!confirm('Are you sure you want to delete ALL grammar check history? This cannot be undone.')) return;
  try {
    const res = await fetch('/api/clear-history', { method: 'POST' });
    const data = await res.json();
    if (res.ok && data.success) {
      showToast('All history cleared.', 'info');
      loadHistoryData();
      loadDashboardData();
    } else {
      showToast(data.error || 'Failed to clear history.', 'error');
    }
  } catch (err) {
    showToast(err.message, 'error');
  }
}

function loadHistoryItemIntoChecker(encodedText) {
  const text = decodeURIComponent(encodedText);
  runSample(text);
}

function showHistoryDetail(id) {
  const item = historyRecords.find(r => r.id === id);
  if (!item) return;

  const modal = document.getElementById('history-modal');
  const content = document.getElementById('modal-content');
  const loadBtn = document.getElementById('modal-load-btn');

  if (content) {
    content.innerHTML = `
      <div style="display: flex; flex-direction: column; gap: 1rem;">
        <div>
          <label style="font-size: 0.8rem; font-weight: 700; color: var(--text-muted); text-transform: uppercase;">Original Sentence</label>
          <div style="padding: 0.75rem; background: var(--bg-secondary); border-radius: var(--radius-sm); margin-top: 0.3rem;">
            ${escapeHtml(item.original_text)}
          </div>
        </div>
        <div>
          <label style="font-size: 0.8rem; font-weight: 700; color: var(--text-muted); text-transform: uppercase;">Corrected Sentence</label>
          <div style="padding: 0.75rem; background: var(--bg-secondary); border-radius: var(--radius-sm); margin-top: 0.3rem; color: var(--success);">
            ${escapeHtml(item.corrected_text)}
          </div>
        </div>
        <div style="display: flex; gap: 1.5rem;">
          <div>
            <label style="font-size: 0.8rem; font-weight: 700; color: var(--text-muted); text-transform: uppercase;">Errors Count</label>
            <p style="font-size: 1.2rem; font-weight: 800; color: var(--danger);">${item.error_count}</p>
          </div>
          <div>
            <label style="font-size: 0.8rem; font-weight: 700; color: var(--text-muted); text-transform: uppercase;">Accuracy</label>
            <p style="font-size: 1.2rem; font-weight: 800; color: var(--primary);">${item.nlp_summary && item.nlp_summary.accuracy !== undefined ? item.nlp_summary.accuracy + '%' : '100%'}</p>
          </div>
          <div>
            <label style="font-size: 0.8rem; font-weight: 700; color: var(--text-muted); text-transform: uppercase;">Timestamp</label>
            <p style="font-size: 0.9rem; color: var(--text-secondary);">${escapeHtml(item.created_at)}</p>
          </div>
        </div>
      </div>
    `;
  }

  if (loadBtn) {
    loadBtn.onclick = () => {
      closeModal();
      loadHistoryItemIntoChecker(encodeURIComponent(item.original_text));
    };
  }

  if (modal) modal.classList.remove('hidden');
}

function closeModal() {
  const modal = document.getElementById('history-modal');
  if (modal) modal.classList.add('hidden');
}

// -------------------------------------------------------------
// 11. Helper Utilities
// -------------------------------------------------------------
function setText(id, text) {
  const el = document.getElementById(id);
  if (el) el.textContent = text;
}

function setBarWidth(id, pct) {
  const el = document.getElementById(id);
  if (el) el.style.width = `${Math.min(100, Math.max(0, pct))}%`;
}

function escapeHtml(str) {
  if (!str) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

function showToast(msg, type = 'info') {
  const container = document.getElementById('toast-container');
  if (!container) return;

  const toast = document.createElement('div');
  toast.className = `toast toast-${type}`;
  toast.innerHTML = `<span>${msg}</span>`;
  container.appendChild(toast);

  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transform = 'translateX(20px)';
    setTimeout(() => toast.remove(), 300);
  }, 3500);
}
