document.getElementById('scan-btn').addEventListener('click', async () => {
    const modelName = document.getElementById('model-select').value;
    const btn = document.getElementById('scan-btn');
    const btnText = document.getElementById('btn-text');
    const spinner = document.getElementById('spinner');
    const resultBox = document.getElementById('result-box');

    // 1. Show loading state immediately
    btn.disabled = true;
    spinner.style.display = 'inline-block';
    btnText.innerText = 'Extracting & Analyzing...';
    resultBox.style.display = 'none';

    const startTime = performance.now();

    try {
        let [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

        // 2. Scrape paragraphs from the active page
        const injectionResults = await chrome.scripting.executeScript({
            target: { tabId: tab.id },
            func: () => {
                const paragraphs = Array.from(document.querySelectorAll('p'))
                    .map(p => p.innerText.trim())
                    .filter(t => t.length > 40);
                return paragraphs.join(' ').substring(0, 15000);
            }
        });

        const articleText = injectionResults[0]?.result || "";

        if (articleText.length < 50) {
            showResult('error', '⚠️ Not enough article text found on this page (found ' + articleText.length + ' chars).');
            return;
        }

        // 3. Call FastAPI directly from the extension popup (bypasses webpage HTTPS blocks)
        const response = await fetch('http://127.0.0.1:8000/api/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: articleText, model: modelName })
        });

        const data = await response.json();
        const elapsedMs = Math.round(performance.now() - startTime);

        if (data.error) {
            showResult('error', `❌ Error: ${data.error}`);
            return;
        }

        const isReal = data.label === "Real";
        const confidence = ((isReal ? data.score_real : 1 - data.score_real) * 100).toFixed(1);

        // 4. Show result right inside the popup
        showResult(
            isReal ? 'real' : 'fake',
            `<div style="font-weight:700; font-size:15px; margin-bottom:6px;">
                ${isReal ? '✅ Likely Authentic' : '🚨 Potential Misinformation'}
             </div>
             <div>Model: <b>${data.model.toUpperCase()}</b></div>
             <div>Confidence: <b>${confidence}%</b></div>
             <div style="font-size:11px; opacity:0.8; margin-top:4px;">Analyzed ${articleText.length} chars in ${elapsedMs} ms</div>`
        );

        // 5. Also inject floating badge on the webpage
        await chrome.scripting.executeScript({
            target: { tabId: tab.id },
            func: injectPageBanner,
            args: [isReal, data.model.toUpperCase(), confidence, elapsedMs]
        });

    } catch (err) {
        showResult('error', `❌ Cannot reach FastAPI server.<br><small>Make sure <b>uvicorn src.api:app --port 8000</b> is running in VS Code.</small>`);
    } finally {
        btn.disabled = false;
        spinner.style.display = 'none';
        btnText.innerText = 'Analyze Current Page';
    }

    function showResult(type, htmlContent) {
        resultBox.className = type;
        resultBox.innerHTML = htmlContent;
        resultBox.style.display = 'block';
    }
});

function injectPageBanner(isReal, modelName, confidence, elapsedMs) {
    const old = document.getElementById('dl-sentinel-overlay');
    if (old) old.remove();

    const banner = document.createElement('div');
    banner.id = 'dl-sentinel-overlay';
    banner.style.cssText = `
        position: fixed; bottom: 24px; right: 24px; z-index: 2147483647;
        width: 300px; padding: 16px; border-radius: 10px;
        background: ${isReal ? '#065f46' : '#9f1239'};
        color: #ffffff; font-family: system-ui, sans-serif;
        box-shadow: 0 10px 25px rgba(0,0,0,0.4);
        border: 2px solid ${isReal ? '#10b981' : '#f43f5e'};
    `;
    banner.innerHTML = `
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
            <span style="font-weight:700; font-size:15px;">
                ${isReal ? '✅ Likely Authentic' : '⚠️ Potential Misinformation'}
            </span>
            <button id="dl-close-btn" style="background:none; border:none; color:white; font-size:18px; cursor:pointer;">×</button>
        </div>
        <div style="font-size:13px;">Model: <b>${modelName}</b> | Confidence: <b>${confidence}%</b></div>
        <div style="font-size:11px; opacity:0.8; margin-top:4px;">Inference time: ${elapsedMs} ms</div>
    `;
    document.body.appendChild(banner);
    document.getElementById('dl-close-btn').onclick = () => banner.remove();
}