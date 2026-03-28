const API_URL = 'http://localhost:5001';
let isProcessing = false;
let currentSessionId = null;
let currentRoomName  = null;
let livePollingTimer = null;
 
// ── Mode switching ────────────────────────────────────────────────────
document.querySelectorAll('.mode-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
 
        const mode = btn.dataset.mode;
        document.querySelectorAll('.mode-panel').forEach(p => p.classList.remove('active'));
        document.getElementById(`${mode}-mode`).classList.add('active');
 
        // Stop live polling when switching away from live mode
        if (mode !== 'live') stopLivePolling();
    });
});
 
// ── File upload handling ──────────────────────────────────────────────
const dropArea  = document.getElementById('drop-area');
const fileInput = document.getElementById('file-input');
const browseBtn = document.getElementById('browse-btn');
const fileInfo  = document.getElementById('file-info');
const fileName  = document.getElementById('file-name');
 
['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    dropArea.addEventListener(eventName, e => { e.preventDefault(); e.stopPropagation(); });
    document.body.addEventListener(eventName, e => { e.preventDefault(); e.stopPropagation(); });
});
 
['dragenter', 'dragover'].forEach(e => dropArea.addEventListener(e, () => dropArea.classList.add('highlight')));
['dragleave', 'drop'].forEach(e => dropArea.addEventListener(e, () => dropArea.classList.remove('highlight')));
 
dropArea.addEventListener('drop', e => handleFiles({ target: { files: e.dataTransfer.files } }));
browseBtn.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', handleFiles);
 
function handleFiles(e) {
    const files = e.target.files;
    if (files.length) {
        fileName.textContent = files[0].name;
        fileInfo.classList.remove('hidden');
    }
}
 
document.getElementById('process-btn')?.addEventListener('click', () => {
    const file = fileInput.files[0];
    if (file) processInterview(file);
});
 
// ── Offline: process interview ────────────────────────────────────────
async function processInterview(file) {
    if (isProcessing) return;
    isProcessing = true;
 
    showResults();
    setTranscriptStatus('⏳ Uploading audio file...');
 
    try {
        // Step 1 — upload
        const formData = new FormData();
        formData.append('audio', file);
        const uploadRes  = await fetch(`${API_URL}/upload`, { method: 'POST', body: formData });
        if (!uploadRes.ok) throw new Error('Upload failed');
        const { audio_path } = await uploadRes.json();
 
        setTranscriptStatus('⏳ Running speaker diarization and transcription...');
 
        // Step 2 — pipeline
        const pipelineRes = await fetch(`${API_URL}/pipeline`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ audio_path }),
        });
        if (!pipelineRes.ok) throw new Error('Pipeline failed');
        const pipelineData = await pipelineRes.json();
        currentSessionId = pipelineData.session_id;
 
        setTranscriptStatus('⏳ Running MedGemma analysis (this takes a minute)...');
 
        // Step 3 — analyze
        const analysisRes = await fetch(`${API_URL}/analyze`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ transcript_path: pipelineData.output_path }),
        });
        if (!analysisRes.ok) throw new Error('Analysis failed');
        const analysisData = await analysisRes.json();
 
        displayTranscript(pipelineData.transcript);
        displayAnalysis(analysisData);
 
    } catch (err) {
        setTranscriptStatus(`❌ Error: ${err.message}`);
        console.error(err);
    } finally {
        isProcessing = false;
    }
}
 
// ── Live: generate REAL tokens ────────────────────────────────────────
document.getElementById('generate-tokens-btn')?.addEventListener('click', generateTokens);
 
async function generateTokens() {
    const btn      = document.getElementById('generate-tokens-btn');
    const original = btn.textContent;
    btn.textContent = '⏳ Generating...';
    btn.disabled    = true;
 
    try {
        const res = await fetch(`${API_URL}/live/tokens`, {
            method:  'POST',
            headers: { 'Content-Type': 'application/json' },
            body:    JSON.stringify({}),
        });
 
        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.error || 'Token generation failed');
        }
 
        const data = await res.json();
        currentRoomName = data.room_name;
 
        document.getElementById('patient-token').value   = data.patient_token;
        document.getElementById('clinician-token').value = data.clinician_token;
 
        // Update the displayed LiveKit URL with the real one from .env
        const urlInput = document.getElementById('livekit-url');
        if (urlInput && data.livekit_url) urlInput.value = data.livekit_url;
 
        // Show room name
        const roomDisplay = document.getElementById('room-name-display');
        if (roomDisplay) {
            roomDisplay.textContent = `Room: ${data.room_name}`;
            roomDisplay.style.display = 'block';
        }
 
        document.getElementById('tokens-display').classList.remove('hidden');
        document.getElementById('stop-interview-btn')?.classList.remove('hidden');
 
        // Start polling for live transcript
        startLivePolling(data.room_name);
 
    } catch (err) {
        alert(`❌ ${err.message}\n\nMake sure LIVEKIT_URL, LIVEKIT_API_KEY, and LIVEKIT_API_SECRET are set in your .env file.`);
    } finally {
        btn.textContent = original;
        btn.disabled    = false;
    }
}
 
// ── Live: poll for transcript updates ─────────────────────────────────
function startLivePolling(roomName) {
    stopLivePolling();
    console.log(`Polling for live transcript: ${roomName}`);
 
    livePollingTimer = setInterval(async () => {
        try {
            const res = await fetch(`${API_URL}/live/transcript?room_name=${encodeURIComponent(roomName)}`);
            if (!res.ok) return;
            const data = await res.json();
 
            if (data.segments && data.segments.length > 0) {
                showResults();
                displayTranscript(data.segments);
            }
        } catch (err) {
            // Polling errors are non-fatal — just log
            console.warn('Live polling error:', err);
        }
    }, 3000); // Poll every 3 seconds
}
 
function stopLivePolling() {
    if (livePollingTimer) {
        clearInterval(livePollingTimer);
        livePollingTimer = null;
    }
}
 
// ── Live: stop interview → index + analyze ────────────────────────────
document.getElementById('stop-interview-btn')?.addEventListener('click', stopInterview);
 
async function stopInterview() {
    if (!currentRoomName) return;
 
    const btn = document.getElementById('stop-interview-btn');
    btn.textContent = '⏳ Processing...';
    btn.disabled    = true;
    stopLivePolling();
 
    setTranscriptStatus('⏳ Indexing live transcript and running MedGemma analysis...');
 
    try {
        const res = await fetch(`${API_URL}/live/stop`, {
            method:  'POST',
            headers: { 'Content-Type': 'application/json' },
            body:    JSON.stringify({ room_name: currentRoomName }),
        });
 
        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.error || 'Stop failed');
        }
 
        const data = await res.json();
        currentSessionId = data.session_id;
 
        displayTranscript(data.transcript);
        displayAnalysis(data);
 
    } catch (err) {
        setTranscriptStatus(`❌ Error: ${err.message}`);
        console.error(err);
    } finally {
        btn.textContent = '⏹ End Interview';
        btn.disabled    = false;
    }
}
 
// ── Copy token buttons ────────────────────────────────────────────────
document.querySelectorAll('.copy-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        const textarea = document.getElementById(btn.dataset.target);
        textarea.select();
        document.execCommand('copy');
        const orig = btn.textContent;
        btn.textContent = 'Copied!';
        setTimeout(() => { btn.textContent = orig; }, 2000);
    });
});
 
// ── Tab switching ─────────────────────────────────────────────────────
document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        const tab = btn.dataset.tab;
        document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
        document.getElementById(`${tab}-tab`).classList.add('active');
    });
});
 
// ── Speaker filter ────────────────────────────────────────────────────
document.querySelectorAll('.filter-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        filterTranscript(btn.dataset.filter);
    });
});
 
// ── Search / Retrieval ────────────────────────────────────────────────
document.getElementById('search-btn')?.addEventListener('click', performSearch);
document.getElementById('search-query')?.addEventListener('keydown', e => {
    if (e.key === 'Enter') performSearch();
});
 
async function performSearch() {
    const query      = document.getElementById('search-query').value.trim();
    const activeMode = document.querySelector('.filter-btn.active')?.dataset.filter || 'all';
    if (!query) return;
 
    const resultsDiv = document.getElementById('search-results');
    resultsDiv.innerHTML = '<p>⏳ Searching...</p>';
 
    try {
        const res = await fetch(`${API_URL}/retrieve`, {
            method:  'POST',
            headers: { 'Content-Type': 'application/json' },
            body:    JSON.stringify({ query, mode: activeMode, k: 5, session_id: currentSessionId }),
        });
        if (!res.ok) throw new Error('Search failed');
        const data = await res.json();
        displaySearchResults(data.results, query);
 
    } catch (err) {
        resultsDiv.innerHTML = `<p>❌ Search failed: ${err.message}</p>`;
    }
}
 
// ── Display: search results with similarity scores ────────────────────
function displaySearchResults(results, query) {
    const resultsDiv = document.getElementById('search-results');
 
    if (!results || results.length === 0) {
        resultsDiv.innerHTML = `<p style="color:#666; padding: 20px;">No results found for "<strong>${query}</strong>"</p>`;
        return;
    }
 
    resultsDiv.innerHTML = `
        <div class="retrieval-header">
            <span class="retrieval-query">Results for: "<strong>${query}</strong>"</span>
            <span class="retrieval-count">${results.length} segments found</span>
        </div>
    `;
 
    results.forEach((result, index) => {
        const role        = (result.role || 'UNKNOWN').toLowerCase();
        const similarity  = result.similarity ?? result.score ?? 0;
        const pct         = Math.round(similarity * 100);
        const startTime   = (result.start_time ?? result.start ?? 0).toFixed(1);
        const endTime     = (result.end_time   ?? result.end   ?? 0).toFixed(1);
 
        // Colour the similarity bar: green > 0.7, amber > 0.4, red below
        const barColor = similarity >= 0.7 ? '#4CAF50' : similarity >= 0.4 ? '#FDBB30' : '#E57373';
 
        const div = document.createElement('div');
        div.className = `result-card ${role}`;
        div.innerHTML = `
            <div class="result-card-header">
                <span class="result-rank">#${index + 1}</span>
                <span class="result-role-badge ${role}">${result.role}</span>
                <span class="result-time">${startTime}s – ${endTime}s</span>
            </div>
            <p class="result-text">${result.text}</p>
            <div class="similarity-row">
                <span class="similarity-label">Similarity</span>
                <div class="similarity-bar-track">
                    <div class="similarity-bar-fill" style="width:${pct}%; background:${barColor};"></div>
                </div>
                <span class="similarity-score" style="color:${barColor};">${similarity.toFixed(3)}</span>
            </div>
        `;
        resultsDiv.appendChild(div);
    });
}
 
// ── Display: transcript ───────────────────────────────────────────────
function displayTranscript(transcript) {
    const content = document.getElementById('transcript-content');
    content.innerHTML = '';
 
    transcript.forEach(line => {
        const role = (line.role || line.speaker || 'UNKNOWN').toLowerCase();
        const start = line.start ?? line.start_time ?? 0;
        const div = document.createElement('div');
        div.className = `transcript-line ${role}`;
        div.dataset.speaker = role;
        div.innerHTML = `
            <span class="timestamp">${Number(start).toFixed(1)}s</span>
            <span class="speaker-label">${(line.role || line.speaker || 'UNKNOWN')}:</span>
            <span class="message-text">${line.text}</span>
        `;
        content.appendChild(div);
    });
}
 
function filterTranscript(filter) {
    document.querySelectorAll('.transcript-line').forEach(line => {
        line.style.display = (filter === 'all' || line.dataset.speaker === filter)
            ? 'block' : 'none';
    });
}
 
// ── Display: analysis ─────────────────────────────────────────────────
function formatMarkdown(text) {
    text = text.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    const lines = text.split('\n');
    let result = '';
    let inList  = false;
 
    lines.forEach(line => {
        const trimmed = line.trim();
        if (trimmed.startsWith('* ') || trimmed.startsWith('- ')) {
            if (!inList) { result += '<ul style="margin-left:20px;margin-top:8px;margin-bottom:8px;">'; inList = true; }
            result += `<li style="margin-bottom:6px;">${trimmed.substring(2)}</li>`;
        } else {
            if (inList) { result += '</ul>'; inList = false; }
            result += trimmed === '' ? '<br>' : `${trimmed}<br>`;
        }
    });
 
    if (inList) result += '</ul>';
    return result;
}
 
function displayAnalysis(analysis) {
    const content = document.getElementById('analysis-content');
    content.innerHTML = `
        <div class="analysis-section">
            <h3>📋 Clinical Summary</h3>
            <p>${formatMarkdown(analysis.summary || '')}</p>
        </div>
        <div class="analysis-section">
            <h3>🩺 Symptom Q&amp;A</h3>
            <p>${formatMarkdown(analysis.symptom_qa || '')}</p>
        </div>
        <div class="analysis-section">
            <h3>📊 Interview Quality</h3>
            <p>${formatMarkdown(analysis.quality || '')}</p>
        </div>
        <div class="analysis-section">
            <h3>🏥 Referral Recommendation</h3>
            <p>${formatMarkdown(analysis.referral || '')}</p>
        </div>
        <div class="analysis-section">
            <h3>❓ Follow-Up Questions</h3>
            <p>${formatMarkdown(analysis.followup || '')}</p>
        </div>
    `;
}
 
// ── Helpers ───────────────────────────────────────────────────────────
function showResults() {
    document.getElementById('results').classList.remove('hidden');
}
 
function setTranscriptStatus(msg) {
    document.getElementById('transcript-content').innerHTML = `<p>${msg}</p>`;
    showResults();
}
