(function () {
  'use strict';

  // Single source of truth is _data/category_colors.yml — landscape.html
  // embeds it as window.CATEGORY_COLORS so this map, the article category
  // tags (see main.scss's own Liquid loop over the same data file), and the
  // legend all draw from the same 15 colors. The literal fallback below only
  // matters if that script tag is ever missing.
  var categoryColors = window.CATEGORY_COLORS || {
    'Architecture': '#3498db',
    'Business': '#e74c3c',
    'Code': '#f39c12',
    'Communication': '#9b59b6',
    'Culture': '#e67e22',
    'Database': '#2ecc71',
    'Dependencies': '#16a085',
    'Management': '#e91e63',
    'Operations': '#34495e',
    'Performance': '#f1c40f',
    'Process': '#27ae60',
    'Requirements': '#8e44ad',
    'Security': '#c0392b',
    'Team': '#1abc9c',
    'Testing': '#ff6b35'
  };

  function colorFor(category) {
    return categoryColors[category] || '#6c757d';
  }

  function hexToRgba(hex, alpha) {
    var match = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    if (!match) return 'rgba(108, 117, 125, ' + alpha + ')';
    var r = parseInt(match[1], 16);
    var g = parseInt(match[2], 16);
    var b = parseInt(match[3], 16);
    return 'rgba(' + r + ', ' + g + ', ' + b + ', ' + alpha + ')';
  }

  var data = window.LANDSCAPE_DATA || { canvas: { width: 1800, height: 1200 }, problems: [], solutions: [] };

  var root = document.querySelector('[data-landscape]');
  if (!root) return;

  var mapWrap = root.querySelector('.landscape__map-wrap');
  var map = root.querySelector('[data-landscape-map]');
  var mapInner = root.querySelector('[data-landscape-map-inner]');
  var article = root.querySelector('[data-landscape-article]');
  var resizer = root.querySelector('[data-landscape-resizer]');
  var legend = root.querySelector('[data-landscape-legend]');
  var legendToggle = root.querySelector('[data-landscape-legend-toggle]');
  var searchInput = root.querySelector('[data-landscape-search]');
  var searchCount = root.querySelector('[data-landscape-search-count]');
  var tabButtons = Array.prototype.slice.call(root.querySelectorAll('[data-landscape-tab]'));

  // MIN_ZOOM sits well below the auto-fit level (typically ~0.35-0.4 for
  // this data), so zooming out still has real headroom beyond the initial
  // view instead of immediately hitting the floor.
  var MIN_ZOOM = 0.08;
  var MAX_ZOOM = 4;

  // Same key analysis-trail.js uses for the speed-nav toggle on regular
  // article pages, so the on/off state carries over either way. This page
  // reimplements the toggle itself (rather than loading analysis-trail.js)
  // because the button only exists once an article has been fetched into
  // `article` below — long after any one-time page-load wiring would run —
  // and because analysis-trail.js's own heading query is scoped to
  // `.page-main-content`, which doesn't exist on this page.
  var speedNavKey = 'problemrider-analysis-trail-speed-nav-v2';

  // Holding space switches into pan mode, same as design tools: while held,
  // dragging anywhere — including over a node label — pans the map instead
  // of selecting it.
  var spacePressed = false;
  var suppressNodeClick = false;

  // Per-tab view/selection state, so switching tabs and coming back keeps
  // pan, zoom and the currently opened article intact.
  var tabState = {
    problems: { zoom: null, panX: 0, panY: 0, activeId: null, nodes: {}, adjusted: false },
    solutions: { zoom: null, panX: 0, panY: 0, activeId: null, nodes: {}, adjusted: false }
  };
  var currentTab = 'problems';

  // window.SITE_BASEURL (from landscape.html, via Liquid's site.baseurl) is
  // "" in local dev but "/problemrider" in production — GitHub Pages serves
  // this project under that path, not domain root. Every Liquid-rendered
  // link already accounts for this via the relative_url filter; this is the
  // one place that builds a URL in plain JS, so it needs the same prefix
  // handed to it explicitly or the fetch 404s in production only.
  var baseurl = window.SITE_BASEURL || '';
  function urlFor(kind, id) {
    return baseurl + '/' + kind + '/' + id + '.html';
  }

  function applyTransform() {
    var state = tabState[currentTab];
    mapInner.style.transform = 'translate(' + state.panX + 'px, ' + state.panY + 'px) scale(' + state.zoom + ')';
  }

  function clampZoom(value) {
    return Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, value));
  }

  // Fits the whole point cloud into the visible viewport. Centers on the
  // actual centroid of the nodes rather than the midpoint of their min/max
  // extents — a skewed layout (denser on one side, with just a few outliers
  // stretching the bounding box the other way) would otherwise center on a
  // point far from where most nodes actually sit, making the map look
  // shifted to one side. The zoom level is likewise based on each node's
  // distance from that centroid, trimmed at the 98th percentile so a
  // handful of far-flung outliers can't force the whole map to zoom out.
  function fitToView(tab) {
    var nodes = data[tab] || [];
    if (!nodes.length) return;
    var meanX = 0, meanY = 0;
    nodes.forEach(function (node) { meanX += node.x; meanY += node.y; });
    meanX /= nodes.length;
    meanY /= nodes.length;

    var devX = nodes.map(function (node) { return Math.abs(node.x - meanX); }).sort(function (a, b) { return a - b; });
    var devY = nodes.map(function (node) { return Math.abs(node.y - meanY); }).sort(function (a, b) { return a - b; });
    var trimmedIndex = Math.min(devX.length - 1, Math.floor(devX.length * 0.98));
    var halfWidth = Math.max(1, devX[trimmedIndex]);
    var halfHeight = Math.max(1, devY[trimmedIndex]);

    var margin = 80;
    var contentWidth = halfWidth * 2 + margin * 2;
    var contentHeight = halfHeight * 2 + margin * 2;
    var viewportWidth = mapWrap.clientWidth || 800;
    var viewportHeight = mapWrap.clientHeight || 600;
    var zoom = clampZoom(Math.min(viewportWidth / contentWidth, viewportHeight / contentHeight, 1));
    var state = tabState[tab];
    state.zoom = zoom;
    state.panX = viewportWidth / 2 - meanX * zoom;
    state.panY = viewportHeight / 2 - meanY * zoom;
  }

  function setActiveNode(tab, id) {
    var state = tabState[tab];
    if (state.activeId && state.nodes[state.activeId]) {
      state.nodes[state.activeId].classList.remove('is-active');
    }
    state.activeId = id;
    if (id && state.nodes[id]) {
      state.nodes[id].classList.add('is-active');
    }
  }

  // The article panel (and its resize handle) only take up space once
  // something is actually selected, so a fresh visit is all map.
  function updateArticleVisibility() {
    var hasSelection = !!tabState[currentTab].activeId;
    article.classList.toggle('is-hidden', !hasSelection);
    resizer.classList.toggle('is-hidden', !hasSelection);
  }

  // The button is freshly cloned in with every article, so it needs its
  // listener re-attached every time rather than once at page load.
  function wireSpeedNavButton() {
    var button = article.querySelector('.speed-nav-toggle');
    if (!button) return;
    button.setAttribute('aria-pressed', String(window.sessionStorage.getItem(speedNavKey) === 'true'));
    button.addEventListener('click', function () {
      var enabled = window.sessionStorage.getItem(speedNavKey) !== 'true';
      window.sessionStorage.setItem(speedNavKey, String(enabled));
      button.setAttribute('aria-pressed', String(enabled));
      button.blur();
    });
  }

  // Mirrors analysis-trail.js's number-key heading navigation, but scoped to
  // the article currently loaded in this panel instead of `.page-main-content`.
  document.addEventListener('keydown', function (event) {
    if (!/^[1-9]$/.test(event.key) || event.ctrlKey || event.metaKey || event.altKey || event.shiftKey) return;
    if (window.sessionStorage.getItem(speedNavKey) !== 'true') return;
    if (/^(INPUT|TEXTAREA|SELECT|BUTTON)$/.test(event.target.tagName) || event.target.isContentEditable) return;
    var headings = Array.prototype.slice.call(article.querySelectorAll('h1, h2, h3'));
    var heading = headings[Number(event.key) - 1];
    if (!heading) return;
    event.preventDefault();
    heading.setAttribute('tabindex', '-1');
    heading.scrollIntoView({ behavior: 'smooth', block: 'start' });
    heading.focus({ preventScroll: true });
  });

  function loadArticle(kind, id) {
    article.innerHTML = '<div class="loading">Loading…</div>';
    window.fetch(urlFor(kind, id)).then(function (response) {
      if (!response.ok) throw new Error('Could not load page');
      return response.text();
    }).then(function (html) {
      var doc = new window.DOMParser().parseFromString(html, 'text/html');
      var loaded = doc.querySelector('.page-main-content article');
      if (!loaded) {
        article.innerHTML = '<div class="error">Could not find the article content.</div>';
        return;
      }
      article.innerHTML = '';
      article.appendChild(loaded.cloneNode(true));
      article.scrollTop = 0;
      wireSpeedNavButton();
    }).catch(function () {
      article.innerHTML = '<div class="error">Could not load this item.</div>';
    });
  }

  // Opening the article shrinks the map viewport (it now shares the row with
  // the article panel), which can leave a node picked while zoomed in
  // sitting outside the new, narrower visible area, or right under the
  // article's edge. Re-centers on that node, at the current zoom level, once
  // the panel's width has actually changed.
  // Full centering, used only the moment the article panel first appears
  // (see selectNode below) — that's the one case where the viewport itself
  // just got narrower out from under whatever was on screen, so the clicked
  // node needs a real re-center or it can end up anywhere, including hidden
  // behind the panel. Subsequent clicks, with the panel already open, don't
  // call this at all: the user is deliberately browsing around the visible
  // area at that point, and forcing a center on every click there felt
  // jarring — a soft "nudge only if hidden" was tried for that case, but a
  // clicked node is so often already comfortably in view that the nudge
  // rarely did anything, which just looked like the feature wasn't there. So
  // now it's this simpler split instead: real center on first open, nothing
  // at all afterwards.
  function centerOnNode(tab, id) {
    var node = (data[tab] || []).filter(function (item) { return item.id === id; })[0];
    var state = tabState[tab];
    if (!node || state.zoom === null) return;
    var viewportWidth = mapWrap.clientWidth || 800;
    var viewportHeight = mapWrap.clientHeight || 600;
    state.panX = viewportWidth / 2 - node.x * state.zoom;
    state.panY = viewportHeight / 2 - node.y * state.zoom;
    state.adjusted = true;
    if (tab === currentTab) applyTransform();
  }

  function selectNode(tab, id, options) {
    options = options || {};
    // Only nudge the view the moment the article panel first appears (and
    // actually changes the map's width) — once it's already open, further
    // clicks just select the new node wherever it is, with no pan changes.
    var articleWasOpen = !!tabState[tab].activeId;
    setActiveNode(tab, id);
    updateArticleVisibility();
    if (!articleWasOpen) centerOnNode(tab, id);
    var kind = tab === 'solutions' ? 'solutions' : 'problems';
    loadArticle(kind, id);
    if (options.updateHash !== false) {
      var hash = '#' + kind + '/' + id;
      if (window.location.hash !== hash) {
        window.history.replaceState(null, '', hash);
      }
    }
  }

  function renderLegend(tab) {
    legend.innerHTML = '';
    // Array.prototype.slice.call only works on array-likes (length + numeric
    // indices) — a Set has neither, so that used to silently yield [].
    var categories = Array.from(new Set((data[tab] || []).map(function (node) { return node.category; }))).sort();
    categories.forEach(function (category) {
      // Reuses the exact same class as a map node's label — a swatch next
      // to plain text was a second, separate way of showing the same color
      // that didn't quite match the pill-shaped, bordered look of the nodes
      // themselves. This way the legend entries and the nodes they're a key
      // for are visibly the same kind of chip, just smaller.
      var chip = document.createElement('span');
      chip.className = 'landscape-node__label landscape__legend-chip';
      chip.textContent = category;
      chip.style.setProperty('--node-color', colorFor(category));
      chip.style.setProperty('--node-bg', hexToRgba(colorFor(category), 0.12));
      legend.appendChild(chip);
    });
  }

  function renderTab(tab) {
    mapInner.innerHTML = '';
    tabState[tab].nodes = {};
    var nodes = data[tab] || [];
    nodes.forEach(function (node) {
      var wrapper = document.createElement('div');
      wrapper.className = 'landscape-node';
      wrapper.style.left = node.x + 'px';
      wrapper.style.top = node.y + 'px';

      var label = document.createElement('button');
      label.type = 'button';
      label.className = 'landscape-node__label';
      label.textContent = node.title;
      label.style.setProperty('--node-color', colorFor(node.category));
      label.style.setProperty('--node-bg', hexToRgba(colorFor(node.category), 0.12));
      label.setAttribute('aria-label', node.title + ' (' + node.category + ')');
      label.addEventListener('click', function (event) {
        event.stopPropagation();
        if (suppressNodeClick) { suppressNodeClick = false; return; }
        selectNode(tab, node.id);
      });

      wrapper.appendChild(label);
      mapInner.appendChild(wrapper);
      tabState[tab].nodes[node.id] = wrapper;
    });
    resolveHeavyOverlaps(tab);
    renderLegend(tab);
    applySearch();
  }

  // create_landscape.py only keeps label *centers* a minimum distance apart,
  // which is a good proxy but not exact — a pair of unusually large
  // (three-line) labels can still end up heavily overlapping. This runs
  // once per render, using each label's actual rendered box (only knowable
  // after it's in the DOM), and nudges apart any pair whose overlap area
  // exceeds half of the smaller label's own area.
  function resolveHeavyOverlaps(tab) {
    var nodes = data[tab] || [];
    var boxes = nodes.map(function (node) {
      var label = tabState[tab].nodes[node.id].querySelector('.landscape-node__label');
      return { node: node, w: label.offsetWidth, h: label.offsetHeight };
    });
    for (var iteration = 0; iteration < 40; iteration++) {
      var moved = false;
      for (var i = 0; i < boxes.length; i++) {
        for (var j = i + 1; j < boxes.length; j++) {
          var a = boxes[i], b = boxes[j];
          var dx = b.node.x - a.node.x;
          var dy = b.node.y - a.node.y;
          var overlapX = (a.w + b.w) / 2 - Math.abs(dx);
          var overlapY = (a.h + b.h) / 2 - Math.abs(dy);
          if (overlapX <= 0 || overlapY <= 0) continue;
          var smallerArea = Math.min(a.w * a.h, b.w * b.h);
          if ((overlapX * overlapY) / smallerArea <= 0.5) continue;
          moved = true;
          // Separate along whichever axis has less overlap to resolve — the
          // shorter push that still clears the >50% threshold.
          if (overlapX < overlapY) {
            var pushX = (overlapX / 2 + 2) * (dx < 0 ? -1 : 1);
            a.node.x -= pushX;
            b.node.x += pushX;
          } else {
            var pushY = (overlapY / 2 + 2) * (dy < 0 ? -1 : 1);
            a.node.y -= pushY;
            b.node.y += pushY;
          }
        }
      }
      if (!moved) break;
    }
    boxes.forEach(function (box) {
      var wrapper = tabState[tab].nodes[box.node.id];
      wrapper.style.left = box.node.x + 'px';
      wrapper.style.top = box.node.y + 'px';
    });
  }

  function switchTab(tab) {
    if (tab === currentTab) return;
    currentTab = tab;
    tabButtons.forEach(function (button) {
      var isActive = button.getAttribute('data-landscape-tab') === tab;
      button.classList.toggle('is-active', isActive);
      button.setAttribute('aria-selected', isActive ? 'true' : 'false');
    });
    // renderTab first: it nudges apart heavily-overlapping labels, and the
    // fit below should frame the map using those corrected positions.
    renderTab(tab);
    if (tabState[tab].zoom === null) fitToView(tab);
    applyTransform();
    if (tabState[tab].activeId) {
      selectNode(tab, tabState[tab].activeId, { updateHash: true });
    } else {
      article.innerHTML = '';
      updateArticleVisibility();
    }
  }

  function applySearch() {
    var query = (searchInput.value || '').trim().toLowerCase();
    var nodes = data[currentTab] || [];
    var matches = 0;
    nodes.forEach(function (node) {
      var element = tabState[currentTab].nodes[node.id];
      if (!element) return;
      if (!query) {
        element.classList.remove('is-dimmed', 'is-matched');
        return;
      }
      var isMatch = node.title.toLowerCase().indexOf(query) !== -1;
      element.classList.toggle('is-matched', isMatch);
      element.classList.toggle('is-dimmed', !isMatch);
      if (isMatch) matches++;
    });
    searchCount.textContent = query ? (matches + ' match' + (matches === 1 ? '' : 'es')) : '';
  }

  // --- Pan & zoom -----------------------------------------------------

  function zoomBy(factor, clientX, clientY) {
    var state = tabState[currentTab];
    var rect = map.getBoundingClientRect();
    var originX = (typeof clientX === 'number' ? clientX : rect.left + rect.width / 2) - rect.left;
    var originY = (typeof clientY === 'number' ? clientY : rect.top + rect.height / 2) - rect.top;
    var newZoom = clampZoom(state.zoom * factor);
    var scaleChange = newZoom / state.zoom;
    state.panX = originX - (originX - state.panX) * scaleChange;
    state.panY = originY - (originY - state.panY) * scaleChange;
    state.zoom = newZoom;
    state.adjusted = true;
    applyTransform();
  }

  map.addEventListener('wheel', function (event) {
    event.preventDefault();
    var factor = event.deltaY < 0 ? 1.12 : 1 / 1.12;
    zoomBy(factor, event.clientX, event.clientY);
  }, { passive: false });

  var panning = false;
  var panStart = null;

  // Two-finger pinch-to-zoom. Pointer Events unify mouse/touch/pen, so a
  // one-finger touch drag already works through the plain panning code
  // below — this layers the second-finger case on top of it. Every active
  // touch is tracked by pointerId (regardless of where it landed, including
  // on a node label — a real pinch can't be expected to keep both fingers
  // off of every label), so a second finger arriving can always take over
  // from a single-finger pan already in progress.
  var activePointers = {};
  var pinch = null;

  function pointerPoints() {
    return Object.keys(activePointers).map(function (id) { return activePointers[id]; });
  }

  map.addEventListener('pointerdown', function (event) {
    activePointers[event.pointerId] = { x: event.clientX, y: event.clientY };
    var pointerCount = Object.keys(activePointers).length;

    if (pointerCount >= 2) {
      try { map.setPointerCapture(event.pointerId); } catch (error) { /* pointer already gone */ }
      panning = false;
      panStart = null;
      var points = pointerPoints().slice(0, 2);
      var rect = map.getBoundingClientRect();
      pinch = {
        startDistance: Math.max(1, Math.hypot(points[0].x - points[1].x, points[0].y - points[1].y)),
        startZoom: tabState[currentTab].zoom,
        startPanX: tabState[currentTab].panX,
        startPanY: tabState[currentTab].panY,
        startMidX: (points[0].x + points[1].x) / 2 - rect.left,
        startMidY: (points[0].y + points[1].y) / 2 - rect.top
      };
      return;
    }

    var onLabel = event.target.closest('.landscape-node__label');
    if (onLabel && !spacePressed) return;
    if (onLabel && spacePressed) suppressNodeClick = true;
    panning = true;
    panStart = { x: event.clientX, y: event.clientY, panX: tabState[currentTab].panX, panY: tabState[currentTab].panY };
    map.classList.add('is-panning');
    map.setPointerCapture(event.pointerId);
  });

  map.addEventListener('pointermove', function (event) {
    if (activePointers[event.pointerId]) {
      activePointers[event.pointerId] = { x: event.clientX, y: event.clientY };
    }

    if (pinch) {
      var points = pointerPoints().slice(0, 2);
      if (points.length < 2) return;
      var state = tabState[currentTab];
      var rect = map.getBoundingClientRect();
      var distance = Math.max(1, Math.hypot(points[0].x - points[1].x, points[0].y - points[1].y));
      var midX = (points[0].x + points[1].x) / 2 - rect.left;
      var midY = (points[0].y + points[1].y) / 2 - rect.top;
      var newZoom = clampZoom(pinch.startZoom * (distance / pinch.startDistance));
      var scaleChange = newZoom / pinch.startZoom;
      // The world point under the pinch's starting midpoint stays under
      // wherever that midpoint has since moved to, at the new zoom — same
      // "zoom around a fixed screen point" math as zoomBy() above, just
      // with that point itself allowed to move (so a two-finger pinch can
      // pan at the same time, same as any map app).
      state.panX = midX - (pinch.startMidX - pinch.startPanX) * scaleChange;
      state.panY = midY - (pinch.startMidY - pinch.startPanY) * scaleChange;
      state.zoom = newZoom;
      state.adjusted = true;
      applyTransform();
      return;
    }

    if (!panning || !panStart) return;
    var panState = tabState[currentTab];
    panState.panX = panStart.panX + (event.clientX - panStart.x);
    panState.panY = panStart.panY + (event.clientY - panStart.y);
    panState.adjusted = true;
    applyTransform();
  });

  function stopPanning(event) {
    if (event && activePointers[event.pointerId] !== undefined) {
      delete activePointers[event.pointerId];
    }
    var remaining = Object.keys(activePointers).length;
    if (remaining < 2) pinch = null;
    if (remaining === 0) {
      panning = false;
      panStart = null;
      map.classList.remove('is-panning');
    }
    // A pan started on top of a node label may or may not still dispatch a
    // native click depending on how the browser resolves pointer capture —
    // clear the flag on the next tick either way, so a real future click on
    // some other node is never swallowed by a stale suppression.
    if (suppressNodeClick) window.setTimeout(function () { suppressNodeClick = false; }, 0);
  }
  map.addEventListener('pointerup', stopPanning);
  map.addEventListener('pointercancel', stopPanning);

  document.addEventListener('keydown', function (event) {
    if (event.code !== 'Space') return;
    // BUTTON is deliberately not excluded here (unlike similar guards
    // elsewhere): almost everything clickable on this page — node labels,
    // tabs, zoom controls — is a button, and it keeps focus after being
    // clicked. Excluding BUTTON meant space stopped engaging pan mode the
    // moment any one of them had been used, which is effectively "always".
    if (/^(INPUT|TEXTAREA|SELECT)$/.test(event.target.tagName) || event.target.isContentEditable) return;
    event.preventDefault();
    if (spacePressed) return;
    spacePressed = true;
    map.classList.add('is-space-panning');
  });

  document.addEventListener('keyup', function (event) {
    if (event.code !== 'Space') return;
    spacePressed = false;
    map.classList.remove('is-space-panning');
  });

  root.querySelector('[data-landscape-zoom-in]').addEventListener('click', function () { zoomBy(1.25); });
  root.querySelector('[data-landscape-zoom-out]').addEventListener('click', function () { zoomBy(1 / 1.25); });
  root.querySelector('[data-landscape-reset]').addEventListener('click', function () {
    fitToView(currentTab);
    tabState[currentTab].adjusted = false;
    applyTransform();
  });

  var legendHiddenKey = 'problemrider-landscape-legend-hidden';
  function setLegendCollapsed(collapsed) {
    legend.classList.toggle('is-collapsed', collapsed);
    legendToggle.setAttribute('aria-expanded', String(!collapsed));
    window.sessionStorage.setItem(legendHiddenKey, String(collapsed));
  }
  setLegendCollapsed(window.sessionStorage.getItem(legendHiddenKey) === 'true');
  legendToggle.addEventListener('click', function () {
    setLegendCollapsed(!legend.classList.contains('is-collapsed'));
  });

  tabButtons.forEach(function (button) {
    button.addEventListener('click', function () {
      var tab = button.getAttribute('data-landscape-tab');
      if (tab === currentTab) {
        // Clicking the already-active tab again closes the open article and
        // returns to the full map view, instead of doing nothing.
        setActiveNode(tab, null);
        article.innerHTML = '';
        updateArticleVisibility();
        if (window.history && window.history.replaceState) {
          window.history.replaceState(null, '', window.location.pathname);
        }
        return;
      }
      switchTab(tab);
    });
  });

  searchInput.addEventListener('input', applySearch);

  // --- Resizable article/map split -------------------------------------

  var splitStorageKey = 'problemrider-landscape-split-v1';
  var savedSplit = window.sessionStorage.getItem(splitStorageKey);
  if (savedSplit) {
    document.documentElement.style.setProperty('--landscape-split', savedSplit + '%');
  }

  var resizing = false;

  resizer.addEventListener('pointerdown', function (event) {
    if (event.button !== 0) return;
    event.preventDefault();
    resizing = true;
    resizer.classList.add('is-dragging');
    resizer.setPointerCapture(event.pointerId);
  });

  resizer.addEventListener('pointermove', function (event) {
    if (!resizing) return;
    var bodyRect = root.querySelector('.landscape__body').getBoundingClientRect();
    var percent = ((event.clientX - bodyRect.left) / bodyRect.width) * 100;
    percent = Math.max(20, Math.min(70, percent));
    document.documentElement.style.setProperty('--landscape-split', percent + '%');
    window.sessionStorage.setItem(splitStorageKey, percent.toFixed(2));
  });

  function stopResizing(event) {
    if (!resizing) return;
    resizing = false;
    resizer.classList.remove('is-dragging');
    if (event && resizer.hasPointerCapture(event.pointerId)) {
      resizer.releasePointerCapture(event.pointerId);
    }
  }
  resizer.addEventListener('pointerup', stopResizing);
  resizer.addEventListener('pointercancel', stopResizing);

  window.addEventListener('resize', function () {
    // Only re-fit when the user has not already interacted with the view,
    // so a manual pan/zoom is never overridden by a window resize.
    if (tabState[currentTab].adjusted) return;
    fitToView(currentTab);
    applyTransform();
  });

  // --- Initial state, including deep links (#problems/slug) ------------

  function initialSelectionFromHash() {
    var hash = window.location.hash.replace(/^#/, '');
    var match = /^(problems|solutions)\/(.+)$/.exec(hash);
    if (!match) return null;
    return { tab: match[1], id: match[2] };
  }

  var initial = initialSelectionFromHash();
  var startTab = initial ? initial.tab : 'problems';

  tabButtons.forEach(function (button) {
    var isActive = button.getAttribute('data-landscape-tab') === startTab;
    button.classList.toggle('is-active', isActive);
    button.setAttribute('aria-selected', isActive ? 'true' : 'false');
  });
  currentTab = startTab;
  renderTab(startTab);
  fitToView(startTab);
  applyTransform();

  if (initial && tabState[startTab].nodes[initial.id]) {
    selectNode(startTab, initial.id, { updateHash: false });
  } else {
    updateArticleVisibility();
  }

  // The Space-to-pan listener is on `document`, so it fires regardless of
  // which element inside the page is focused — but not if focus is still
  // outside the page entirely (e.g. the browser's own address bar, right
  // after typing/pasting the URL and hitting enter, before ever clicking
  // into the page). Proactively pulling focus onto the map — once up front,
  // and again the moment the pointer actually enters it — closes that gap
  // without needing an explicit click first.
  if (map.focus) map.focus({ preventScroll: true });
  map.addEventListener('pointerenter', function () {
    if (document.activeElement !== map) map.focus({ preventScroll: true });
  });
})();
