/* ============================================================
   Object Detection Guide — interactive labs (vanilla JS, no deps)
   A page opts in with  <div class="lab" data-lab="iou|nms|map|grid|letterbox|atlas"></div>
   Every lab mirrors a function in code/odlab so the numbers agree.
   ============================================================ */
(function () {
  'use strict';

  var C = {
    gt: '#22c55e', pred: '#f59e0b', enc: '#06b6d4', red: '#ef4444', dim: '#555', text: '#e8e8e8',
    grid: 'rgba(255,255,255,0.08)', purple: '#a855f7', pink: '#ec4899', blue: '#3b82f6'
  };

  function el(tag, attrs, html) {
    var e = document.createElement(tag);
    if (attrs) for (var k in attrs) e.setAttribute(k, attrs[k]);
    if (html !== undefined) e.innerHTML = html;
    return e;
  }
  function canvas(w, h) {
    var c = el('canvas');
    var r = window.devicePixelRatio || 1;
    c.width = w * r; c.height = h * r; c.style.width = w + 'px'; c.style.height = h + 'px';
    c.getContext('2d').setTransform(r, 0, 0, r, 0, 0);
    c._w = w; c._h = h;
    return c;
  }
  function pos(c, ev) {
    var b = c.getBoundingClientRect();
    var t = ev.touches ? ev.touches[0] : ev;
    return { x: (t.clientX - b.left) * c._w / b.width, y: (t.clientY - b.top) * c._h / b.height };
  }
  function frame(root, title) {
    root.innerHTML = '';
    root.appendChild(el('div', { 'class': 'lab-title' }, title));
    var body = el('div', { 'class': 'lab-body' });
    root.appendChild(body);
    return body;
  }
  function fmt(x, d) { return (x === null || x === undefined || isNaN(x)) ? '—' : x.toFixed(d === undefined ? 3 : d); }

  // ---------------------------------------------------------------- box math (mirrors odlab/boxes.py)
  var B = {
    area: function (b) { return Math.max(0, b[2] - b[0]) * Math.max(0, b[3] - b[1]); },
    inter: function (a, b) {
      var w = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0]));
      var h = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
      return w * h;
    },
    iou: function (a, b) { var i = B.inter(a, b); return i / (B.area(a) + B.area(b) - i + 1e-9); },
    enc: function (a, b) { return [Math.min(a[0], b[0]), Math.min(a[1], b[1]), Math.max(a[2], b[2]), Math.max(a[3], b[3])]; },
    rho2: function (a, b) {
      var dx = (a[0] + a[2] - b[0] - b[2]) / 2, dy = (a[1] + a[3] - b[1] - b[3]) / 2;
      return dx * dx + dy * dy;
    },
    all: function (a, b) {
      var i = B.inter(a, b), u = B.area(a) + B.area(b) - i, iou = i / (u + 1e-9);
      var e = B.enc(a, b), cw = e[2] - e[0], ch = e[3] - e[1], cA = cw * ch, c2 = cw * cw + ch * ch + 1e-9;
      var giou = iou - (cA - u) / (cA + 1e-9);
      var r2 = B.rho2(a, b), diou = iou - r2 / c2;
      var w1 = a[2] - a[0], h1 = a[3] - a[1], w2 = b[2] - b[0], h2 = b[3] - b[1];
      var v = 4 / (Math.PI * Math.PI) * Math.pow(Math.atan(w2 / h2) - Math.atan(w1 / h1), 2);
      var alpha = v / (1 - iou + v + 1e-9);
      var ciou = diou - alpha * v;
      var eiou = diou - (w1 - w2) * (w1 - w2) / (cw * cw + 1e-9) - (h1 - h2) * (h1 - h2) / (ch * ch + 1e-9);
      var w2d = Math.pow((a[0] + a[2] - b[0] - b[2]) / 2, 2) + Math.pow((a[1] + a[3] - b[1] - b[3]) / 2, 2) +
        Math.pow((w1 - w2) / 2, 2) + Math.pow((h1 - h2) / 2, 2);
      var nwd = Math.exp(-Math.sqrt(w2d) / 12.8);
      return { iou: iou, giou: giou, diou: diou, ciou: ciou, eiou: eiou, nwd: nwd, inter: i, union: u, enc: e, v: v, alpha: alpha, rho2: r2, c2: c2 };
    }
  };

  function strokeBox(ctx, b, color, dash, lw) {
    ctx.save();
    ctx.strokeStyle = color; ctx.lineWidth = lw || 2;
    if (dash) ctx.setLineDash(dash);
    ctx.strokeRect(b[0], b[1], b[2] - b[0], b[3] - b[1]);
    ctx.restore();
  }
  function fillBox(ctx, b, color) { ctx.fillStyle = color; ctx.fillRect(b[0], b[1], b[2] - b[0], b[3] - b[1]); }
  function label(ctx, txt, x, y, color) {
    ctx.save(); ctx.font = '11px JetBrains Mono, monospace'; ctx.fillStyle = color || C.text; ctx.fillText(txt, x, y); ctx.restore();
  }

  // Draggable / resizable boxes helper
  function dragBoxes(c, boxes, redraw) {
    var active = null, mode = null, off = null;
    function hit(p) {
      for (var i = boxes.length - 1; i >= 0; i--) {
        var b = boxes[i].b;
        if (Math.abs(p.x - b[2]) < 9 && Math.abs(p.y - b[3]) < 9) return [i, 'resize'];
      }
      for (var j = boxes.length - 1; j >= 0; j--) {
        var bb = boxes[j].b;
        if (p.x > bb[0] && p.x < bb[2] && p.y > bb[1] && p.y < bb[3]) return [j, 'move'];
      }
      return null;
    }
    function down(ev) {
      var p = pos(c, ev), h = hit(p);
      if (!h) return;
      active = h[0]; mode = h[1];
      off = { x: p.x - boxes[active].b[0], y: p.y - boxes[active].b[1] };
      ev.preventDefault();
    }
    function move(ev) {
      var p = pos(c, ev);
      if (active === null) {
        var h = hit(p);
        c.style.cursor = h ? (h[1] === 'resize' ? 'nwse-resize' : 'move') : 'crosshair';
        return;
      }
      var b = boxes[active].b;
      if (mode === 'move') {
        var w = b[2] - b[0], hh = b[3] - b[1];
        b[0] = Math.max(0, Math.min(c._w - w, p.x - off.x)); b[1] = Math.max(0, Math.min(c._h - hh, p.y - off.y));
        b[2] = b[0] + w; b[3] = b[1] + hh;
      } else {
        b[2] = Math.max(b[0] + 4, Math.min(c._w, p.x)); b[3] = Math.max(b[1] + 4, Math.min(c._h, p.y));
      }
      redraw(); ev.preventDefault();
    }
    function up() { active = null; }
    c.addEventListener('mousedown', down); c.addEventListener('touchstart', down, { passive: false });
    window.addEventListener('mousemove', move); c.addEventListener('touchmove', move, { passive: false });
    window.addEventListener('mouseup', up); window.addEventListener('touchend', up);
  }

  // ---------------------------------------------------------------- IoU playground
  function labIoU(root) {
    var body = frame(root, 'Lab — IoU family playground (drag boxes, drag the corner handle to resize)');
    var c = canvas(440, 300), ctx = c.getContext('2d');
    body.appendChild(c);
    var panel = el('div', { 'class': 'lab-panel' });
    var out = el('div', { 'class': 'lab-readout' });
    panel.appendChild(out);
    panel.appendChild(el('div', { 'class': 'lab-hint' }, 'Green = ground truth, amber = prediction, cyan dashed = smallest enclosing box C. ' +
      'Pull the boxes apart: IoU sticks at 0 (no gradient), GIoU/DIoU/CIoU keep falling — that is why they train better.'));
    body.appendChild(panel);
    var boxes = [{ b: [80, 70, 220, 200] }, { b: [170, 120, 330, 240] }];
    function redraw() {
      var g = boxes[0].b, p = boxes[1].b, m = B.all(p, g);
      ctx.clearRect(0, 0, c._w, c._h);
      strokeBox(ctx, m.enc, C.enc, [5, 4], 1);
      var ix = [Math.max(g[0], p[0]), Math.max(g[1], p[1]), Math.min(g[2], p[2]), Math.min(g[3], p[3])];
      if (ix[2] > ix[0] && ix[3] > ix[1]) fillBox(ctx, ix, 'rgba(245,158,11,0.18)');
      strokeBox(ctx, g, C.gt); strokeBox(ctx, p, C.pred);
      ctx.save(); ctx.strokeStyle = C.pink; ctx.setLineDash([2, 3]); ctx.beginPath();
      ctx.moveTo((g[0] + g[2]) / 2, (g[1] + g[3]) / 2); ctx.lineTo((p[0] + p[2]) / 2, (p[1] + p[3]) / 2); ctx.stroke(); ctx.restore();
      [g, p].forEach(function (b, i) { fillBox(ctx, [b[2] - 4, b[3] - 4, b[2] + 4, b[3] + 4], i ? C.pred : C.gt); });
      label(ctx, 'gt', g[0] + 4, g[1] + 13, C.gt); label(ctx, 'pred', p[0] + 4, p[1] + 13, C.pred);
      out.textContent =
        'IoU   = ' + fmt(m.iou) + '   (inter ' + Math.round(m.inter) + ' / union ' + Math.round(m.union) + ')\n' +
        'GIoU  = ' + fmt(m.giou) + '   IoU - |C \\ U| / |C|\n' +
        'DIoU  = ' + fmt(m.diou) + '   IoU - rho^2/c^2 = IoU - ' + fmt(m.rho2 / m.c2) + '\n' +
        'CIoU  = ' + fmt(m.ciou) + '   DIoU - alpha*v  (v=' + fmt(m.v, 4) + ')\n' +
        'EIoU  = ' + fmt(m.eiou) + '\n' +
        'NWD   = ' + fmt(m.nwd) + '   (C = 12.8 px)\n\n' +
        'losses (1 - x):  IoU ' + fmt(1 - m.iou) + '  GIoU ' + fmt(1 - m.giou) + '  CIoU ' + fmt(1 - m.ciou);
    }
    dragBoxes(c, boxes, redraw);
    redraw();
  }

  // ---------------------------------------------------------------- NMS stepper
  function labNMS(root) {
    var body = frame(root, 'Lab — NMS, step by step');
    var c = canvas(440, 300), ctx = c.getContext('2d');
    body.appendChild(c);
    var panel = el('div', { 'class': 'lab-panel' });
    var ctr = el('div', { 'class': 'lab-controls' });
    var thr = el('input', { type: 'range', min: '0.1', max: '0.9', step: '0.05', value: '0.5' });
    var thrL = el('label', null, 'IoU thr '); thrL.appendChild(thr); var thrV = el('span', null, '0.50'); thrL.appendChild(thrV);
    var mode = el('select'); ['greedy', 'soft-nms (gaussian)', 'diou-nms'].forEach(function (m) { mode.appendChild(el('option', null, m)); });
    var bStep = el('button', null, 'Step'), bRun = el('button', null, 'Run all'), bReset = el('button', null, 'Reset'), bShuffle = el('button', null, 'New scene');
    ctr.appendChild(thrL); ctr.appendChild(mode); ctr.appendChild(bStep); ctr.appendChild(bRun); ctr.appendChild(bReset); ctr.appendChild(bShuffle);
    var out = el('div', { 'class': 'lab-readout' });
    panel.appendChild(ctr); panel.appendChild(out);
    panel.appendChild(el('div', { 'class': 'lab-hint' }, 'Two overlapping people + a few singles. Watch the two-person cluster: greedy NMS at a low threshold deletes the second person; Soft-NMS only lowers its score; DIoU-NMS spares it because the centres are far apart.'));
    body.appendChild(panel);
    var seed = 3, dets, state;
    function rnd() { seed = (seed * 16807) % 2147483647; return (seed - 1) / 2147483646; }
    function scene() {
      dets = [];
      var objs = [[60, 60, 150, 250], [125, 70, 215, 255], [270, 40, 400, 140], [290, 180, 380, 280]];
      objs.forEach(function (o, k) {
        var n = 3 + Math.floor(rnd() * 3);
        for (var i = 0; i < n; i++) {
          var j = function (s) { return (rnd() - 0.5) * s; };
          var w = o[2] - o[0], h = o[3] - o[1];
          dets.push({ b: [o[0] + j(w * 0.15), o[1] + j(h * 0.15), o[2] + j(w * 0.15), o[3] + j(h * 0.15)], s: +(0.35 + 0.6 * rnd() - (k === 1 ? 0.1 : 0)).toFixed(2), obj: k });
        }
      });
      reset();
    }
    function reset() {
      state = { remaining: dets.map(function (d, i) { return i; }), kept: [], removed: [], score: dets.map(function (d) { return d.s; }), current: null };
      draw();
    }
    function step() {
      var t = parseFloat(thr.value), m = mode.value;
      var rem = state.remaining.filter(function (i) { return state.score[i] > 0.05; });
      if (!rem.length) { state.current = null; draw(); return false; }
      rem.sort(function (a, b) { return state.score[b] - state.score[a]; });
      var top = rem[0];
      state.kept.push(top); state.current = top;
      var rest = rem.slice(1), keepRest = [];
      rest.forEach(function (i) {
        var iou = B.iou(dets[top].b, dets[i].b);
        if (m.indexOf('soft') === 0) {
          state.score[i] = state.score[i] * Math.exp(-iou * iou / 0.5);
          if (state.score[i] > 0.05) keepRest.push(i); else state.removed.push(i);
        } else {
          var crit = m === 'diou-nms' ? B.all(dets[top].b, dets[i].b).diou : iou;
          if (crit > t) state.removed.push(i); else keepRest.push(i);
        }
      });
      state.remaining = keepRest;
      draw();
      return true;
    }
    function draw() {
      ctx.clearRect(0, 0, c._w, c._h);
      dets.forEach(function (d, i) {
        var col = C.dim, dash = [3, 3];
        if (state.kept.indexOf(i) >= 0) { col = C.gt; dash = null; }
        else if (state.removed.indexOf(i) >= 0) { col = 'rgba(239,68,68,0.55)'; dash = [2, 4]; }
        if (i === state.current) col = C.pred;
        strokeBox(ctx, d.b, col, dash, i === state.current ? 3 : 1.5);
        if (state.removed.indexOf(i) < 0) label(ctx, state.score[i].toFixed(2), d.b[0] + 3, d.b[1] + 12, col);
      });
      out.textContent = 'boxes: ' + dets.length + '   kept: ' + state.kept.length + '   suppressed: ' + state.removed.length +
        '   pending: ' + state.remaining.length + '\nkept scores: ' + state.kept.map(function (i) { return state.score[i].toFixed(2); }).join(', ') +
        '\npairwise IoU evaluations (greedy worst case n(n-1)/2): ' + (dets.length * (dets.length - 1) / 2);
    }
    thr.addEventListener('input', function () { thrV.textContent = (+thr.value).toFixed(2); reset(); });
    mode.addEventListener('change', reset);
    bStep.addEventListener('click', step);
    bRun.addEventListener('click', function () { var g = 0; while (step() && g++ < 200) { } });
    bReset.addEventListener('click', reset);
    bShuffle.addEventListener('click', function () { seed = Math.floor(Math.random() * 1e6) + 1; scene(); });
    scene();
  }

  // ---------------------------------------------------------------- AP builder
  function labMAP(root) {
    var body = frame(root, 'Lab — build a PR curve and compute AP three ways (click rows to flip TP/FP)');
    var panel = el('div', { 'class': 'lab-panel' });
    var ctr = el('div', { 'class': 'lab-controls' });
    var ngt = el('input', { type: 'number', min: '1', max: '50', value: '6', style: 'width:4em' });
    var l1 = el('label', null, '# ground truth '); l1.appendChild(ngt);
    var bAdd = el('button', null, 'Add detection'), bRand = el('button', null, 'Randomise');
    ctr.appendChild(l1); ctr.appendChild(bAdd); ctr.appendChild(bRand);
    panel.appendChild(ctr);
    var tblWrap = el('div', { style: 'max-height:240px;overflow:auto' });
    panel.appendChild(tblWrap);
    var c = canvas(300, 240), ctx = c.getContext('2d');
    var right = el('div', { 'class': 'lab-panel' });
    right.appendChild(c);
    var out = el('div', { 'class': 'lab-readout' });
    right.appendChild(out);
    right.appendChild(el('div', { 'class': 'lab-hint' }, 'Grey = raw precision, amber = monotone envelope. COCO reads the envelope at 101 recall points; VOC07 at 11; VOC10+ integrates it exactly.'));
    body.appendChild(panel); body.appendChild(right);
    var dets = [[0.95, 1], [0.91, 1], [0.84, 0], [0.80, 1], [0.71, 0], [0.66, 1], [0.52, 0], [0.43, 1], [0.30, 0]];
    function compute() {
      var n = Math.max(1, parseInt(ngt.value, 10) || 1);
      var d = dets.slice().sort(function (a, b) { return b[0] - a[0]; });
      var tp = 0, fp = 0, R = [], P = [];
      d.forEach(function (x) { if (x[1]) tp++; else fp++; R.push(Math.min(tp / n, 1)); P.push(tp / (tp + fp)); });
      var env = P.slice();
      for (var i = env.length - 2; i >= 0; i--) env[i] = Math.max(env[i], env[i + 1]);
      // all points
      var r = [0].concat(R, [1]), p = [0].concat(P, [0]), pe = p.slice();
      for (var k = pe.length - 2; k >= 0; k--) pe[k] = Math.max(pe[k], pe[k + 1]);
      var allp = 0; for (var j = 0; j < r.length - 1; j++) if (r[j + 1] !== r[j]) allp += (r[j + 1] - r[j]) * pe[j + 1];
      var v11 = 0; for (var t = 0; t <= 10; t++) { var mx = 0; for (var q = 0; q < r.length; q++) if (r[q] >= t / 10 - 1e-12) mx = Math.max(mx, p[q]); v11 += mx / 11; }
      var c101 = 0; for (var s = 0; s <= 100; s++) { var th = s / 100, idx = -1; for (var z = 0; z < R.length; z++) if (R[z] >= th - 1e-12) { idx = z; break; } if (idx >= 0) c101 += env[idx] / 101; }
      return { d: d, R: R, P: P, env: env, allp: allp, v11: v11, c101: c101, n: n, tp: tp };
    }
    function render() {
      var m = compute();
      var html = '<table><thead><tr><th>#</th><th>score</th><th>TP?</th><th>P</th><th>R</th></tr></thead><tbody>';
      m.d.forEach(function (x, i) {
        html += '<tr data-i="' + i + '" style="cursor:pointer"><td>' + (i + 1) + '</td><td>' + x[0].toFixed(2) + '</td><td style="color:' + (x[1] ? C.gt : C.red) + '">' + (x[1] ? 'TP' : 'FP') + '</td><td>' + m.P[i].toFixed(2) + '</td><td>' + m.R[i].toFixed(2) + '</td></tr>';
      });
      tblWrap.innerHTML = html + '</tbody></table>';
      Array.prototype.forEach.call(tblWrap.querySelectorAll('tr[data-i]'), function (tr) {
        tr.addEventListener('click', function () {
          var i = +tr.getAttribute('data-i'); var sorted = m.d[i];
          dets.forEach(function (x) { if (x === sorted) x[1] = 1 - x[1]; });
          render();
        });
      });
      // plot
      var W = c._w, H = c._h, L = 34, T = 10, PW = W - L - 10, PH = H - T - 28;
      ctx.clearRect(0, 0, W, H);
      ctx.strokeStyle = C.grid; ctx.lineWidth = 1;
      for (var g = 0; g <= 4; g++) { ctx.beginPath(); ctx.moveTo(L, T + PH * g / 4); ctx.lineTo(L + PW, T + PH * g / 4); ctx.stroke(); }
      label(ctx, 'precision', 2, T + 8, '#999'); label(ctx, 'recall', L + PW - 40, H - 6, '#999');
      label(ctx, '1', 22, T + 8, '#777'); label(ctx, '0', 22, T + PH, '#777');
      function X(r) { return L + r * PW; } function Y(p) { return T + (1 - p) * PH; }
      ctx.fillStyle = 'rgba(245,158,11,0.12)'; ctx.beginPath(); ctx.moveTo(X(0), Y(0));
      var px = 0;
      m.R.forEach(function (r, i) { ctx.lineTo(X(px), Y(m.env[i])); ctx.lineTo(X(r), Y(m.env[i])); px = r; });
      ctx.lineTo(X(px), Y(0)); ctx.closePath(); ctx.fill();
      ctx.strokeStyle = C.pred; ctx.lineWidth = 2; ctx.beginPath(); px = 0;
      m.R.forEach(function (r, i) { if (i === 0) ctx.moveTo(X(0), Y(m.env[0])); ctx.lineTo(X(px), Y(m.env[i])); ctx.lineTo(X(r), Y(m.env[i])); px = r; });
      ctx.stroke();
      ctx.fillStyle = '#999';
      m.R.forEach(function (r, i) { ctx.beginPath(); ctx.arc(X(r), Y(m.P[i]), 3, 0, 6.3); ctx.fill(); });
      out.textContent = 'TP ' + m.tp + ' / GT ' + m.n + '   final recall ' + fmt(m.R[m.R.length - 1] || 0, 2) + '\n' +
        'AP (VOC10+ all-points) = ' + fmt(m.allp) + '\nAP (VOC07 11-point)    = ' + fmt(m.v11) + '\nAP (COCO 101-point)    = ' + fmt(m.c101) +
        '\n\nCOCO AP = mean of this over 10 IoU thresholds x all classes.';
    }
    bAdd.addEventListener('click', function () { dets.push([+(Math.random() * 0.9 + 0.05).toFixed(2), Math.random() > 0.5 ? 1 : 0]); render(); });
    bRand.addEventListener('click', function () { dets = []; for (var i = 0; i < 10; i++) dets.push([+(Math.random()).toFixed(2), Math.random() > 0.45 ? 1 : 0]); render(); });
    ngt.addEventListener('input', render);
    render();
  }

  // ---------------------------------------------------------------- grid / assignment explorer
  function labGrid(root) {
    var body = frame(root, 'Lab — which grid points can see this object? (drag/resize the box)');
    var c = canvas(320, 320), ctx = c.getContext('2d');
    body.appendChild(c);
    var panel = el('div', { 'class': 'lab-panel' });
    var ctr = el('div', { 'class': 'lab-controls' });
    var stride = el('select'); [8, 16, 32].forEach(function (s) { stride.appendChild(el('option', { value: s }, 'stride ' + s)); });
    stride.value = '16';
    var rule = el('select'); ['centre inside box (FCOS/TAL)', 'centre radius 2.5 strides (SimOTA)', 'YOLOv5 cell + 2 neighbours'].forEach(function (r) { rule.appendChild(el('option', null, r)); });
    var stal = el('input', { type: 'checkbox' }); var stalL = el('label', null, ''); stalL.appendChild(stal); stalL.appendChild(document.createTextNode(' STAL floor (16 px)'));
    ctr.appendChild(stride); ctr.appendChild(rule); ctr.appendChild(stalL);
    var out = el('div', { 'class': 'lab-readout' });
    panel.appendChild(ctr); panel.appendChild(out);
    panel.appendChild(el('div', { 'class': 'lab-hint' }, 'Shrink the box below the stride: at stride 32 a 20 px object can fall between grid centres and get zero candidates. That is the small-object problem in one picture — and what P2 heads, centre radius and STAL each fix differently.'));
    body.appendChild(panel);
    var boxes = [{ b: [100, 110, 180, 170] }];
    function redraw() {
      var s = +stride.value, b = boxes[0].b.slice(), r = rule.value;
      if (stal.checked) {
        var cx = (b[0] + b[2]) / 2, cy = (b[1] + b[3]) / 2, w = Math.max(16, b[2] - b[0]), h = Math.max(16, b[3] - b[1]);
        b = [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2];
      }
      ctx.clearRect(0, 0, c._w, c._h);
      ctx.strokeStyle = C.grid;
      for (var x = 0; x <= c._w; x += s) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, c._h); ctx.stroke(); }
      for (var y = 0; y <= c._h; y += s) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(c._w, y); ctx.stroke(); }
      if (stal.checked) strokeBox(ctx, b, C.purple, [3, 3], 1);
      strokeBox(ctx, boxes[0].b, C.gt, null, 2);
      fillBox(ctx, [boxes[0].b[2] - 4, boxes[0].b[3] - 4, boxes[0].b[2] + 4, boxes[0].b[3] + 4], C.gt);
      var ocx = (boxes[0].b[0] + boxes[0].b[2]) / 2, ocy = (boxes[0].b[1] + boxes[0].b[3]) / 2;
      var n = 0;
      for (var gy = 0; gy < c._h / s; gy++) for (var gx = 0; gx < c._w / s; gx++) {
        var px = (gx + 0.5) * s, py = (gy + 0.5) * s, on = false;
        if (r.indexOf('SimOTA') >= 0) {
          var rad = 2.5 * s;
          on = (px > b[0] && px < b[2] && py > b[1] && py < b[3]) || (Math.abs(px - ocx) < rad && Math.abs(py - ocy) < rad);
        } else if (r.indexOf('YOLOv5') >= 0) {
          var cgx = Math.floor(ocx / s), cgy = Math.floor(ocy / s), fx = ocx / s - cgx, fy = ocy / s - cgy;
          on = (gx === cgx && gy === cgy) || (gy === cgy && gx === cgx + (fx < 0.5 ? -1 : 1)) || (gx === cgx && gy === cgy + (fy < 0.5 ? -1 : 1));
        } else {
          on = px > b[0] && px < b[2] && py > b[1] && py < b[3];
        }
        ctx.fillStyle = on ? C.pred : 'rgba(255,255,255,0.18)';
        ctx.beginPath(); ctx.arc(px, py, on ? 3.5 : 1.5, 0, 6.3); ctx.fill();
        if (on) n++;
      }
      var w0 = boxes[0].b[2] - boxes[0].b[0], h0 = boxes[0].b[3] - boxes[0].b[1];
      out.textContent = 'object ' + Math.round(w0) + ' x ' + Math.round(h0) + ' px  (area ' + Math.round(w0 * h0) + ', COCO ' + (w0 * h0 < 1024 ? 'small' : w0 * h0 < 9216 ? 'medium' : 'large') + ')\n' +
        'stride ' + s + ' -> grid ' + (c._w / s) + ' x ' + (c._h / s) + '\ncandidate points: ' + n + (n === 0 ? '   <-- no positive possible!' : '') +
        '\n(TAL then keeps the top-10 candidates by s^0.5 * IoU^6)';
    }
    [stride, rule, stal].forEach(function (x) { x.addEventListener('change', redraw); });
    dragBoxes(c, boxes, redraw);
    redraw();
  }

  // ---------------------------------------------------------------- letterbox calculator
  function labLetterbox(root) {
    var body = frame(root, 'Lab — letterbox calculator');
    var panel = el('div', { 'class': 'lab-panel' });
    var ctr = el('div', { 'class': 'lab-controls' });
    function num(lbl, v) { var i = el('input', { type: 'number', value: v, style: 'width:5em' }); var l = el('label', null, lbl + ' '); l.appendChild(i); ctr.appendChild(l); return i; }
    var W = num('W', 1920), H = num('H', 1080), S = num('target', 640);
    var auto = el('input', { type: 'checkbox' }); var al = el('label', null, ''); al.appendChild(auto); al.appendChild(document.createTextNode(' auto (stride-32 min rect)')); ctr.appendChild(al);
    var bx = num('box x1', 800), by = num('y1', 400);
    panel.appendChild(ctr);
    var out = el('div', { 'class': 'lab-readout' }); panel.appendChild(out);
    var c = canvas(300, 300), ctx = c.getContext('2d');
    body.appendChild(c); body.appendChild(panel);
    function run() {
      var w = +W.value, h = +H.value, s = +S.value;
      var r = Math.min(s / h, s / w), uw = Math.round(w * r), uh = Math.round(h * r);
      var dw = s - uw, dh = s - uh;
      if (auto.checked) { dw = dw % 32; dh = dh % 32; }
      dw /= 2; dh /= 2;
      var top = Math.round(dh - 0.1), bottom = Math.round(dh + 0.1), left = Math.round(dw - 0.1), right = Math.round(dw + 0.1);
      var ow = uw + left + right, oh = uh + top + bottom;
      var x1 = +bx.value * r + left, y1 = +by.value * r + top;
      out.textContent = 'scale r = min(' + s + '/' + h + ', ' + s + '/' + w + ') = ' + r.toFixed(4) +
        '\nresized  ' + uw + ' x ' + uh + '\npadding  left ' + left + ', right ' + right + ', top ' + top + ', bottom ' + bottom +
        '\noutput   ' + ow + ' x ' + oh + '  (' + (100 * (1 - uw * uh / (ow * oh))).toFixed(1) + '% grey pixels)' +
        '\n\npoint (' + bx.value + ', ' + by.value + ') -> (' + x1.toFixed(1) + ', ' + y1.toFixed(1) + ')' +
        '\ninverse: x = (x\' - ' + left + ') / ' + r.toFixed(4) + ',  y = (y\' - ' + top + ') / ' + r.toFixed(4);
      var k = 280 / Math.max(ow, oh);
      ctx.clearRect(0, 0, 300, 300);
      fillBox(ctx, [10, 10, 10 + ow * k, 10 + oh * k], '#2a2a2a');
      fillBox(ctx, [10 + left * k, 10 + top * k, 10 + (left + uw) * k, 10 + (top + uh) * k], 'rgba(59,130,246,0.35)');
      ctx.fillStyle = C.pred; ctx.beginPath(); ctx.arc(10 + x1 * k, 10 + y1 * k, 4, 0, 6.3); ctx.fill();
      label(ctx, 'grey = pad (114)', 14, 24, '#999');
    }
    [W, H, S, auto, bx, by].forEach(function (x) { x.addEventListener('input', run); x.addEventListener('change', run); });
    run();
  }

  // ---------------------------------------------------------------- Detection Atlas
  var FAMILY_COLORS = ['#f59e0b', '#22c55e', '#3b82f6', '#ec4899', '#a855f7', '#06b6d4', '#ef4444', '#14b8a6', '#eab308', '#f97316', '#84cc16', '#e879f9', '#38bdf8', '#fb7185', '#c084fc', '#4ade80', '#fbbf24', '#60a5fa'];
  function labAtlas(root) {
    var body = frame(root, 'Detection Atlas — COCO AP vs latency (T4 TensorRT FP16 rows only on the plot)');
    body.style.display = 'block';
    var url = (window.OD_BASEURL || '') + '/assets/data/detectors.json';
    var status = el('div', { 'class': 'lab-hint' }, 'Loading ' + url + ' …');
    body.appendChild(status);
    fetch(url).then(function (r) { return r.json(); }).then(function (data) {
      status.remove();
      build(body, data.models);
    }).catch(function () { status.textContent = 'Could not load detectors.json (open this page through the published site, not the raw GitHub view).'; });
  }
  function build(body, models) {
    var fams = Array.from(new Set(models.map(function (m) { return m.family; })));
    var col = {}; fams.forEach(function (f, i) { col[f] = FAMILY_COLORS[i % FAMILY_COLORS.length]; });
    var filt = el('div', { 'class': 'atlas-filters lab-controls' });
    var tier = el('select'); ['all tiers', 'mcu', 'tiny', 'edge', 'realtime', 'large', 'open-vocab', 'mllm', 'classic'].forEach(function (t) { tier.appendChild(el('option', null, t)); });
    var lic = el('select'); ['any license', 'permissive only (Apache/MIT/BSD)', 'copyleft (AGPL/GPL)'].forEach(function (t) { lic.appendChild(el('option', null, t)); });
    var nmsSel = el('select'); ['NMS: any', 'NMS-free capable', 'needs NMS'].forEach(function (t) { nmsSel.appendChild(el('option', null, t)); });
    var q = el('input', { type: 'text', placeholder: 'search name / family / org', style: 'font-size:0.78rem;padding:0.15rem 0.4rem;background:#1a1a1a;color:#e8e8e8;border:1px solid #2a2a2a;border-radius:4px' });
    var maxLat = el('input', { type: 'range', min: '1', max: '20', step: '0.5', value: '20' }); var ml = el('label', null, 'max latency '); ml.appendChild(maxLat); var mlv = el('span', null, '20 ms'); ml.appendChild(mlv);
    [tier, lic, nmsSel, q, ml].forEach(function (x) { filt.appendChild(x); });
    body.appendChild(filt);
    var c = canvas(Math.min(760, (body.clientWidth || 760) - 4), 360), ctx = c.getContext('2d');
    body.appendChild(c);
    var legend = el('div', { 'class': 'atlas-legend' });
    body.appendChild(legend);
    var tip = el('div', { 'class': 'lab-readout', style: 'min-height:2.4em;margin:0.5rem 0' }, 'Hover a point.');
    body.appendChild(tip);
    var wrap = el('div', { 'class': 'atlas-table-wrap' });
    body.appendChild(wrap);
    var sortKey = 'coco_ap', sortDir = -1, pts = [];
    function permissive(l) { return /Apache|MIT|BSD/.test(l || ''); }
    function copyleft(l) { return /GPL/.test(l || ''); }
    function pass(m) {
      var t = tier.value; if (t !== 'all tiers' && m.tier !== t) return false;
      if (lic.value.indexOf('permissive') === 0 && !permissive(m.license)) return false;
      if (lic.value.indexOf('copyleft') === 0 && !copyleft(m.license)) return false;
      if (nmsSel.value === 'NMS-free capable' && m.nms === true) return false;
      if (nmsSel.value === 'needs NMS' && m.nms !== true) return false;
      var s = q.value.toLowerCase(); if (s && (m.name + ' ' + m.family + ' ' + m.org).toLowerCase().indexOf(s) < 0) return false;
      return true;
    }
    function draw() {
      mlv.textContent = maxLat.value + ' ms';
      var rows = models.filter(pass);
      var plot = rows.filter(function (m) { return m.latency_hw === 'T4 TRT FP16' && m.coco_ap && m.latency_ms && m.latency_ms <= +maxLat.value; });
      var W = c._w, H = c._h, L = 44, T = 12, PW = W - L - 14, PH = H - T - 34;
      var xmax = Math.max(4, +maxLat.value), ymin = 20, ymax = 62;
      function X(v) { return L + v / xmax * PW; } function Y(v) { return T + (1 - (v - ymin) / (ymax - ymin)) * PH; }
      ctx.clearRect(0, 0, W, H);
      ctx.strokeStyle = C.grid; ctx.fillStyle = '#777'; ctx.font = '10px JetBrains Mono, monospace';
      for (var a = 25; a <= 60; a += 5) { ctx.beginPath(); ctx.moveTo(L, Y(a)); ctx.lineTo(L + PW, Y(a)); ctx.stroke(); ctx.fillText(a, 18, Y(a) + 3); }
      for (var l = 0; l <= xmax; l += (xmax > 10 ? 2 : 1)) { ctx.beginPath(); ctx.moveTo(X(l), T); ctx.lineTo(X(l), T + PH); ctx.stroke(); ctx.fillText(l, X(l) - 3, T + PH + 14); }
      ctx.fillStyle = '#999'; ctx.fillText('COCO val AP', 2, 10); ctx.fillText('latency, ms (T4, TensorRT, FP16, batch 1)', L + PW / 2 - 110, H - 4);
      // family lines
      var byFam = {}; plot.forEach(function (m) { (byFam[m.family] = byFam[m.family] || []).push(m); });
      Object.keys(byFam).forEach(function (f) {
        var arr = byFam[f].slice().sort(function (a, b) { return a.latency_ms - b.latency_ms; });
        ctx.strokeStyle = col[f]; ctx.globalAlpha = 0.5; ctx.lineWidth = 1.2; ctx.beginPath();
        arr.forEach(function (m, i) { if (i) ctx.lineTo(X(m.latency_ms), Y(m.coco_ap)); else ctx.moveTo(X(m.latency_ms), Y(m.coco_ap)); });
        ctx.stroke(); ctx.globalAlpha = 1;
      });
      pts = [];
      plot.forEach(function (m) {
        var x = X(m.latency_ms), y = Y(m.coco_ap);
        ctx.fillStyle = col[m.family]; ctx.beginPath(); ctx.arc(x, y, /O365|Objects365|DINOv|VFM|PE-Core/.test(m.pretrain || '') ? 4.5 : 3.5, 0, 6.3); ctx.fill();
        if (/O365|Objects365|DINOv|VFM|PE-Core/.test(m.pretrain || '')) { ctx.strokeStyle = '#fff'; ctx.lineWidth = 1; ctx.stroke(); }
        pts.push({ x: x, y: y, m: m });
      });
      legend.innerHTML = Object.keys(byFam).map(function (f) { return '<span style="--dot:' + col[f] + '">' + f + '</span>'; }).join('') +
        '<span style="--dot:#fff">white ring = extra pre-training (Objects365 / foundation backbone)</span>';
      // table
      rows.sort(function (a, b) { var x = a[sortKey], y = b[sortKey]; if (x === null || x === undefined) return 1; if (y === null || y === undefined) return -1; return (x > y ? 1 : x < y ? -1 : 0) * sortDir; });
      var cols = [['name', 'Model'], ['family', 'Family'], ['year', 'Year'], ['params_m', 'Params M'], ['gflops', 'GFLOPs'], ['input', 'Input'], ['coco_ap', 'COCO AP'], ['latency_ms', 'Lat. ms'], ['latency_hw', 'Latency HW'], ['pretrain', 'Pre-training'], ['license', 'License'], ['nms', 'NMS']];
      var h = '<table><thead><tr>' + cols.map(function (k) { return '<th data-k="' + k[0] + '">' + k[1] + (sortKey === k[0] ? (sortDir > 0 ? ' ▲' : ' ▼') : '') + '</th>'; }).join('') + '</tr></thead><tbody>';
      rows.forEach(function (m) {
        h += '<tr>' + cols.map(function (k) {
          var v = m[k[0]];
          if (k[0] === 'nms') v = v === 'opt' ? 'optional' : v ? 'yes' : 'no';
          if (k[0] === 'license') { var cls = permissive(v) ? 'apache' : copyleft(v) ? 'agpl' : 'nc'; v = '<span class="badge ' + cls + '">' + v + '</span>'; }
          return '<td>' + (v === null || v === undefined ? '—' : v) + (k[0] === 'name' && m.note ? ' <span title="' + m.note.replace(/"/g, '&quot;') + '" style="color:#777">ⓘ</span>' : '') + '</td>';
        }).join('') + '</tr>';
      });
      wrap.innerHTML = h + '</tbody></table>';
      Array.prototype.forEach.call(wrap.querySelectorAll('th'), function (th) {
        th.addEventListener('click', function () { var k = th.getAttribute('data-k'); if (k === sortKey) sortDir = -sortDir; else { sortKey = k; sortDir = -1; } draw(); });
      });
    }
    c.addEventListener('mousemove', function (ev) {
      var p = pos(c, ev), best = null, bd = 100;
      pts.forEach(function (q2) { var d = Math.hypot(q2.x - p.x, q2.y - p.y); if (d < bd) { bd = d; best = q2; } });
      if (best && bd < 12) {
        var m = best.m;
        tip.textContent = m.name + '  —  ' + m.coco_ap + ' AP @ ' + m.latency_ms + ' ms, ' + (m.params_m || '?') + ' M params, ' + m.input + ' px\n' +
          'pre-training: ' + m.pretrain + '   license: ' + m.license + '   NMS: ' + (m.nms === 'opt' ? 'optional' : m.nms ? 'required' : 'none') + (m.note ? '\nnote: ' + m.note : '') + '\nsource: ' + m.source;
      }
    });
    [tier, lic, nmsSel].forEach(function (x) { x.addEventListener('change', draw); });
    [q, maxLat].forEach(function (x) { x.addEventListener('input', draw); });
    draw();
  }

  var LABS = { iou: labIoU, nms: labNMS, map: labMAP, grid: labGrid, letterbox: labLetterbox, atlas: labAtlas };
  function init() {
    Array.prototype.forEach.call(document.querySelectorAll('.lab[data-lab]'), function (root) {
      var f = LABS[root.getAttribute('data-lab')];
      if (f) { try { f(root); } catch (e) { root.textContent = 'Lab failed to load: ' + e; } }
    });
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init();
})();
