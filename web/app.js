const API_BASE = 'http://localhost:8000/api';

// Routing
function handleRoute() {
    const hash = window.location.hash || '#/';
    const route = hash.replace('#', '');
    
    // Update active nav
    document.querySelectorAll('.nav-links a').forEach(a => {
        a.classList.remove('active');
        if (a.getAttribute('data-route') === route) a.classList.add('active');
    });

    // Show correct page
    document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
    
    if (route === '/') document.getElementById('page-home').classList.add('active');
    else if (route === '/chat') document.getElementById('page-chat').classList.add('active');
    else if (route === '/eval') document.getElementById('page-eval').classList.add('active');
    else if (route === '/health') document.getElementById('page-health').classList.add('active');
}

window.addEventListener('hashchange', handleRoute);
handleRoute();

// Update Budget UI
function updateBudget(usage) {
    if (usage && usage.budget_remaining_calls !== undefined) {
        document.getElementById('budget-counter').innerText = `${usage.budget_remaining_calls}`;
    }
}

// 1. Ingest
document.getElementById('btn-ingest').addEventListener('click', async () => {
    const btn = document.getElementById('btn-ingest');
    const resBox = document.getElementById('ingest-result');
    
    btn.innerText = 'Indexing...';
    resBox.classList.add('hidden');
    
    try {
        const res = await fetch(`${API_BASE}/ingest`, { method: 'POST' });
        const data = await res.json();
        resBox.innerText = JSON.stringify(data, null, 2);
        resBox.classList.remove('hidden');
    } catch (e) {
        resBox.innerText = `Error: ${e.message}`;
        resBox.classList.remove('hidden');
    } finally {
        btn.innerText = 'Rebuild Index';
    }
});

// 2. Chat
const chatInput = document.getElementById('chat-input');
const chatHistory = document.getElementById('chat-history');
const btnSend = document.getElementById('btn-send');
const citationsPanel = document.getElementById('citations-panel');
const citationsList = document.getElementById('citations-list');

function addMessage(text, sender) {
    const div = document.createElement('div');
    div.className = `message ${sender}`;
    div.innerText = text;
    chatHistory.appendChild(div);
    chatHistory.scrollTop = chatHistory.scrollHeight;
}

async function sendQuery() {
    const text = chatInput.value.trim();
    if (!text) return;
    
    addMessage(text, 'user');
    chatInput.value = '';
    
    const isAgent = document.getElementById('agent-toggle').checked;
    
    try {
        const res = await fetch(`${API_BASE}/query`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                question: text,
                mode: isAgent ? 'agent' : 'simple'
            })
        });
        
        const data = await res.json();
        
        if (res.status === 429) {
            addMessage("Error: Budget Exceeded (Max LLM calls reached).", 'bot');
            return;
        }
        
        addMessage(data.answer, 'bot');
        updateBudget(data.usage);
        
        // Citations
        if (data.citations && data.citations.length > 0) {
            citationsList.innerHTML = '';
            data.citations.forEach(cit => {
                const li = document.createElement('li');
                li.innerHTML = `<strong>[${cit.id}] ${cit.source}</strong><br>${cit.snippet}...`;
                citationsList.appendChild(li);
            });
            citationsPanel.classList.remove('hidden');
        } else {
            citationsPanel.classList.add('hidden');
        }
        
    } catch (e) {
        addMessage(`Error: ${e.message}`, 'bot');
    }
}

btnSend.addEventListener('click', sendQuery);
chatInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') sendQuery();
});

// 3. Eval
document.getElementById('btn-eval').addEventListener('click', async () => {
    const btn = document.getElementById('btn-eval');
    const grid = document.getElementById('eval-metrics');
    
    btn.innerText = 'Running Evaluation...';
    
    try {
        const res = await fetch(`${API_BASE}/eval`, { method: 'POST' });
        const data = await res.json();
        
        if (data.averages) {
            document.getElementById('score-faithfulness').innerText = data.averages.faithfulness.toFixed(2);
            document.getElementById('score-relevancy').innerText = data.averages.relevancy.toFixed(2);
            document.getElementById('score-precision').innerText = data.averages.context_precision.toFixed(2);
            document.getElementById('score-recall').innerText = data.averages.context_recall.toFixed(2);
            grid.classList.remove('hidden');
        }
    } catch (e) {
        alert("Evaluation failed: " + e.message);
    } finally {
        btn.innerText = 'Run Golden Set Eval';
    }
});

// 4. Health
document.getElementById('btn-health-parse').addEventListener('click', async () => {
    const btn = document.getElementById('btn-health-parse');
    const fileInput = document.getElementById('health-file');
    const resBox = document.getElementById('health-result');
    
    if (!fileInput.files[0]) {
        alert("Please select a PDF file");
        return;
    }
    
    btn.innerText = 'Processing...';
    resBox.classList.add('hidden');
    
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    
    try {
        const res = await fetch(`${API_BASE}/health/parse`, {
            method: 'POST',
            body: formData
        });
        const data = await res.json();
        resBox.innerText = JSON.stringify(data, null, 2);
        resBox.classList.remove('hidden');
    } catch (e) {
        resBox.innerText = `Error: ${e.message}`;
        resBox.classList.remove('hidden');
    } finally {
        btn.innerText = 'Parse & Index';
    }
});
