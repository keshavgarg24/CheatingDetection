async function fetchJSON(url) {
  const res = await fetch(url);
  return await res.json();
}

function setTable(tbody, rows, cols) {
  tbody.innerHTML = '';
  rows.forEach(r => {
    const tr = document.createElement('tr');
    cols.forEach(c => {
      const td = document.createElement('td');
      td.textContent = r[c] ?? '';
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });
}

async function refresh() {
  try {
    const status = await fetchJSON('/api/status');
    const viols = await fetchJSON('/api/violations');
    setTable(document.querySelector('#statusTable tbody'), status, ['student_id', 'strikes', 'status', 'last_update']);
    setTable(document.querySelector('#violTable tbody'), viols, ['student_id', 'type', 'detail', 'ts']);
  } catch (e) {
    console.error(e);
  }
}

setInterval(refresh, 2000);
refresh();
