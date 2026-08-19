// static/app.js
(() => {
  // Socket.IO client (assumes socket.io client script is loaded on the page)
  const socket = io();

  // Utility: safe text setter
  function setText(id, text) {
    const el = document.getElementById(id);
    if (el) el.innerText = text;
  }

  // Utility: create element with attrs
  function el(tag, attrs = {}, text = "") {
    const e = document.createElement(tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (text) e.textContent = text;
    return e;
  }

  // Table update handler (keeps existing UI updated)
  socket.on('table_update', (state) => {
    try {
      setText('potAmt', state.pot ?? 0);
      setText('gen', (state.generation ?? 0).toString());
      const bf = (state.best_fitness ?? 0);
      setText('bf', (typeof bf === 'number') ? bf.toFixed(3) : String(bf));
      if (state.players && Array.isArray(state.players)) {
        for (let i = 0; i < 9; i++) {
          const elId = 's' + i;
          const seatEl = document.getElementById(elId);
          if (seatEl && state.players[i]) {
            seatEl.innerText = `Seat ${i} - ${state.players[i].stack}`;
            const avatar = document.querySelector(`#seat${i} .avatar`) || document.querySelector(`#seat${i} img.avatar`);
            if (avatar && state.players[i].avatar) avatar.src = state.players[i].avatar;
          }
        }
      }
    } catch (e) {
      console.error('table_update handler error', e);
    }
  });

  // GA update handler (more detailed stats)
  socket.on('ga_update', (stats) => {
    try {
      // stats may contain generation, best_fitness, mean_fitness, top_std
      if (stats) {
        if (stats.generation !== undefined) setText('gen', String(stats.generation));
        if (stats.best_fitness !== undefined) {
          const bf = Number(stats.best_fitness);
          setText('bf', isFinite(bf) ? bf.toFixed(3) : String(stats.best_fitness));
        }
        // Optionally show mean and std in info area
        const infoEl = document.getElementById('info');
        if (infoEl) {
          const mean = (stats.mean_fitness !== undefined) ? ` | Mean: ${Number(stats.mean_fitness).toFixed(3)}` : '';
          const std = (stats.top_std !== undefined) ? ` | TopStd: ${Number(stats.top_std).toFixed(3)}` : '';
          // keep generation and best fitness already shown; append extras
          const genText = `Gen: ${stats.generation ?? '0'} | Best fitness: ${ (stats.best_fitness !== undefined) ? Number(stats.best_fitness).toFixed(3) : '0' }`;
          infoEl.innerText = genText + mean + std;
        }
      }
    } catch (e) {
      console.error('ga_update handler error', e);
    }
  });

  // Hall of Fame update handler: renders list and matrix
  socket.on('hof_update', (data) => {
    try {
      const listEl = document.getElementById('hofList');
      const matrixEl = document.getElementById('hofMatrix');
      if (!listEl || !matrixEl) return;

      // Labels
      const labels = (data && data.labels) ? data.labels : [];
      const matrix = (data && data.matrix) ? data.matrix : [];

      if (!labels.length) {
        listEl.innerText = "No HOF yet";
        matrixEl.innerHTML = "";
        return;
      }

      // Render label list
      listEl.innerText = labels.join(', ');

      // Render numeric table (compact)
      const n = labels.length;
      // compute min/max for color scaling
      let minv = Infinity, maxv = -Infinity;
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          const v = Number(matrix[i] && matrix[i][j] ? matrix[i][j] : 0);
          if (isFinite(v)) {
            if (v < minv) minv = v;
            if (v > maxv) maxv = v;
          }
        }
      }
      if (!isFinite(minv)) { minv = 0; maxv = 1; }
      if (minv === maxv) { maxv = minv + 1; }

      // Build HTML table with color-coded cells
      const tbl = document.createElement('table');
      tbl.style.borderCollapse = 'collapse';
      tbl.style.width = '100%';
      tbl.style.fontSize = '12px';

      // header row
      const header = document.createElement('tr');
      header.appendChild(el('th', { style: 'padding:2px; text-align:left; background:rgba(255,255,255,0.04);' }, ''));
      for (let j = 0; j < n; j++) {
        header.appendChild(el('th', { style: 'padding:2px; text-align:right; background:rgba(255,255,255,0.04);' }, labels[j]));
      }
      tbl.appendChild(header);

      // rows
      for (let i = 0; i < n; i++) {
        const row = document.createElement('tr');
        row.appendChild(el('td', { style: 'padding:2px; font-weight:bold; background:rgba(255,255,255,0.02);' }, labels[i]));
        for (let j = 0; j < n; j++) {
          const v = Number(matrix[i] && matrix[i][j] ? matrix[i][j] : 0);
          // normalize to 0..1
          const t = (v - minv) / (maxv - minv);
          
          // color: blue (low) -> white (mid) -> orange (high)
          const r = Math.round(255 * Math.min(1, Math.max(0, 0.9 * t + 0.1)));
          const g = Math.round(255 * Math.min(1, Math.max(0, 0.9 * (1 - Math.abs(t - 0.5)) + 0.1)));
          const b = Math.round(255 * Math.min(1, Math.max(0, 1 - t)));
          const bg = `rgb(${r},${g},${b})`;
          const cell = el('td', { style: `padding:2px; text-align:right; background:${bg}; color:#000;` }, isFinite(v) ? v.toFixed(2) : '0.00');
          row.appendChild(cell);
        }
        tbl.appendChild(row);
      }

      // replace content
      matrixEl.innerHTML = "";
      matrixEl.appendChild(tbl);

    } catch (e) {
      console.error('hof_update handler error', e);
    }
  });

  // Optional: request initial HOF and stats on connect (server also emits on connect)
  socket.on('connect', () => {
    console.log('Connected to server via Socket.IO');

    // server already emits table_update and hof_update on connect; this is just a safety ping
    socket.emit('client_ready', {});
  });

  // Deal button handler (if present)
  const dealBtn = document.getElementById('dealBtn');
  if (dealBtn) {
    dealBtn.addEventListener('click', () => {
      socket.emit('deal', {});
    });
  }

  // Expose for debugging
  window.__pokerEvolverSocket = socket;
})();
