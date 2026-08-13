(function () {
  'use strict';

  // Versioned storage prevents edges created by an older navigation-based
  // implementation from being mixed with the semantic causal graph.
  var storageKey = 'problemrider-analysis-trail-v12';
  var pendingKey = 'problemrider-analysis-trail-pending-edge-v12';
  var expandedKey = 'problemrider-analysis-trail-expanded-v1';
  var historyKey = 'problemrider-analysis-trail-history-v1';
  var speedNavKey = 'problemrider-analysis-trail-speed-nav-v2';
  var focusKey = 'problemrider-analysis-trail-focus-v1';
  var maxNodes = 200;
  var maxEdges = 30;
  var namespace = 'http://www.w3.org/2000/svg';
  var spacePressed = false;
  var pendingAdd = null;
  // Node id the user started a connection from, or null when not connecting.
  var linkingFrom = null;

  function stopLinking() {
    linkingFrom = null;
    var graph = document.querySelector('[data-analysis-trail-graph] svg');
    if (graph) graph.classList.remove('is-linking');
    document.querySelectorAll('.analysis-trail__node-group.is-link-source').forEach(function (group) {
      group.classList.remove('is-link-source');
    });
  }

  // The node the user currently looks at. Kept outside the trail order and
  // persisted, so a full page load starts with exactly one active node instead
  // of whatever node happened to be added last.
  var focusedNodeId = null;
  try { focusedNodeId = window.sessionStorage.getItem(focusKey); } catch (error) { focusedNodeId = null; }
  // Node whose reference menu and controls are currently open, kept outside the
  // render scope so an asynchronous re-render can restore it.
  var menuOpenNodeId = null;

  function setFocusedNodeId(nodeId) {
    focusedNodeId = nodeId || null;
    try {
      if (focusedNodeId) window.sessionStorage.setItem(focusKey, focusedNodeId);
      else window.sessionStorage.removeItem(focusKey);
    } catch (error) { /* session storage is optional */ }
  }

  // The article is replaced in place, so nothing resets the scroll offset by
  // itself. Every possible scroll container is reset because the layout scrolls
  // the window in the wide layout and the content column in the narrow one.
  function scrollArticleToTop() {
    var pageContent = document.querySelector('.page-content');
    if (pageContent) pageContent.scrollTop = 0;
    var pageMain = document.querySelector('.page-main-content');
    if (pageMain) pageMain.scrollTop = 0;
    if (document.scrollingElement) document.scrollingElement.scrollTop = 0;
    document.documentElement.scrollTop = 0;
    document.body.scrollTop = 0;
    window.scrollTo(0, 0);
  }

  var currentAddType = 'cause';

  function addTypeUsesSolutionCatalog(type) {
    return type === 'solution' || type === 'similar-solution';
  }

  // Every type stays available on every node, because an entry can play both
  // roles. Only the preselection follows the node the "+" was used on.
  function updateTypeButtonsDOM() {
    var buttons = document.querySelectorAll('.analysis-trail__type-btn');
    buttons.forEach(function (btn) {
      btn.hidden = false;
      if (btn.getAttribute('data-type') === currentAddType) {
        btn.classList.add('is-active');
      } else {
        btn.classList.remove('is-active');
      }
    });
  }

  function openAddModal(sourceNode, position) {
    var modal = document.querySelector('[data-analysis-trail-add-modal]');
    var text = document.querySelector('[data-analysis-trail-add-modal-text]');
    var results = document.querySelector('[data-analysis-trail-add-search-results]');
    var selection = document.querySelector('[data-analysis-trail-add-selection]');
    if (!modal || !text) return;
    pendingAdd = { sourceNode: sourceNode, position: { x: position.x, y: position.y }, selected: null };
    var isSolutionSource = sourceNode.id.indexOf('solution:') === 0 || sourceNode.type === 'solution';
    currentAddType = isSolutionSource ? 'similar-solution' : 'cause';
    updateTypeButtonsDOM();
    text.value = '';
    if (results) { results.innerHTML = ''; results.hidden = true; }
    if (selection) selection.textContent = 'Custom node will be created.';
    modal.hidden = false;
    text.focus();
  }

  function getTrail() {
    try {
      var saved = JSON.parse(window.sessionStorage.getItem(storageKey));
      if (saved && Array.isArray(saved.nodes) && Array.isArray(saved.edges)) {
        saved.positions = saved.positions || {};
        saved.pan = saved.pan || { x: 0, y: 300 };
        return saved;
      }
      return { nodes: [], edges: [], positions: {}, pan: { x: 0, y: 300 } };
    } catch (error) {
      return { nodes: [], edges: [], positions: {}, pan: { x: 0, y: 300 } };
    }
  }

  function saveTrail(trail) {
    window.sessionStorage.setItem(storageKey, JSON.stringify(trail));
    updateMenuCount();
  }

  function updateMenuCount() {
    var badge = document.querySelector('[data-analysis-trail-count]');
    if (!badge) return;
    try {
      var saved = JSON.parse(window.sessionStorage.getItem(storageKey));
      var count = saved && Array.isArray(saved.nodes) ? saved.nodes.length : 0;
      badge.textContent = String(count);
      badge.hidden = count === 0;
    } catch (error) { badge.hidden = true; }
  }

  function snapshot(trail) {
    return JSON.parse(JSON.stringify(trail));
  }

  function getHistory() {
    try {
      var saved = JSON.parse(window.sessionStorage.getItem(historyKey));
      return saved && Array.isArray(saved.undo) && Array.isArray(saved.redo) ? saved : { undo: [], redo: [] };
    } catch (error) {
      return { undo: [], redo: [] };
    }
  }

  function saveHistory(history) {
    window.sessionStorage.setItem(historyKey, JSON.stringify(history));
  }

  function rememberChange(trail) {
    var history = getHistory();
    history.undo.push(snapshot(trail));
    if (history.undo.length > 30) history.undo.shift();
    history.redo = [];
    saveHistory(history);
  }

  function edgeForLink(link) {
    var element = link;
    while (element && element.tagName !== 'ARTICLE') {
      var previous = element.previousElementSibling;
      while (previous) {
        if (previous.tagName === 'H2') {
          var heading = previous.textContent.trim();
          if (heading.indexOf('Symptoms') === 0 || heading.indexOf('Symptom') === 0) return { label: 'causes', direction: 'reverse', targetType: 'symptom' };
          if (heading.indexOf('Root Causes') === 0 || heading.indexOf('Causes') === 0) return { label: 'causes', direction: 'forward', targetType: 'root cause' };
          break;
        }
        previous = previous.previousElementSibling;
      }
      element = element.parentElement;
    }

    // A solution always points to the problem it addresses, even if the user
    // navigated from that problem to the solution.
    if (link.closest('.related-solutions')) return { label: 'addresses', direction: 'reverse', targetType: 'solution' };
    if (link.closest('.addressed-problems')) return { label: 'addresses', direction: 'forward', targetType: 'problem' };
    if (link.closest('.related-problems')) return null;
    return null;
  }

  function currentNode() {
    var article = document.querySelector('[data-analysis-node]');
    if (!article) return null;
    return {
      id: article.getAttribute('data-analysis-node-id'),
      title: article.getAttribute('data-analysis-node-title'),
      type: article.getAttribute('data-analysis-node-type'),
      url: window.location.pathname + window.location.search
    };
  }

  function addCurrentNode(trail, node) {
    if (!node) return;
    var existingIndex = trail.nodes.findIndex(function (item) { return item.id === node.id; });
    var existing = existingIndex === -1 ? null : trail.nodes[existingIndex];
    if (existing) {
      existing.url = node.url;
      existing.title = node.title;
      existing.type = node.type;
      trail.nodes.splice(existingIndex, 1);
      trail.nodes.push(existing);
      return;
    }
    trail.nodes.push(node);
    if (trail.nodes.length > maxNodes) {
      var removed = trail.nodes.shift();
      trail.edges = trail.edges.filter(function (edge) { return edge.from !== removed.id && edge.to !== removed.id; });
      delete trail.positions[removed.id];
    }
  }

  function addPendingEdge(trail, node) {
    var pending;
    try { pending = JSON.parse(window.sessionStorage.getItem(pendingKey)); } catch (error) { pending = null; }
    window.sessionStorage.removeItem(pendingKey);
    if (!pending || !node || pending.from === node.id) return;
    if (!trail.nodes.some(function (item) { return item.id === pending.from; })) return;
    if (pending.targetType) {
      var targetNode = trail.nodes.filter(function (item) { return item.id === node.id; })[0];
      if (targetNode) {
        var previousType = targetNode.type;
        targetNode.type = pending.targetType;
        // A node reached through a semantic reference must use the matching
        // type row (symptoms at the top, causes at the bottom). Do not carry
        // over the temporary position it had while loading as a problem.
        if (previousType !== pending.targetType) delete trail.positions[node.id];
      }
    }
    var from = pending.direction === 'reverse' ? node.id : pending.from;
    var to = pending.direction === 'reverse' ? pending.from : node.id;
    var exists = trail.edges.some(function (edge) {
      return edge.from === from && edge.to === to && edge.label === pending.label;
    });
    if (!exists) trail.edges.push({ from: from, to: to, label: pending.label });
    if (trail.edges.length > maxEdges) trail.edges.shift();
  }

  function svgElement(name, attributes) {
    var element = document.createElementNS(namespace, name);
    Object.keys(attributes || {}).forEach(function (key) { element.setAttribute(key, attributes[key]); });
    return element;
  }

  function localReferences(pageDocument, kind) {
    var links = [];
    var selector = kind === 'solutions' ? '.related-solutions a[href]' :
      (kind === 'addressed-problems' ? '.addressed-problems a[href]' :
        (kind === 'similar-solutions' ? '.analysis-trail-similar-solutions a[href]' :
          (kind === 'similar-problems' ? '.related-problems a[href]' : '')));
    if (selector) {
      links = Array.prototype.slice.call(pageDocument.querySelectorAll(selector));
    } else {
      var headingStart = kind === 'symptoms' ? 'Symptoms' : 'Causes';
      var heading = Array.prototype.slice.call(pageDocument.querySelectorAll('h2')).filter(function (item) {
        var text = item.textContent.trim();
        return kind === 'causes' ? /^(Root )?Causes/.test(text) : /^Symptoms?/.test(text);
      })[0];
      if (heading) {
        var sibling = heading.nextElementSibling;
        while (sibling && sibling.tagName !== 'H2') {
          links = links.concat(Array.prototype.slice.call(sibling.querySelectorAll('a[href]')));
          sibling = sibling.nextElementSibling;
        }
      }
    }
    var seen = {};
    return links.map(function (link) {
      var url = new URL(link.href, window.location.href);
      return { title: link.textContent.trim(), url: url.pathname + url.search };
    }).filter(function (item) {
      return /\/(problems|solutions)\/[^/]+\.html$/.test(item.url) && item.title && !seen[item.url] && (seen[item.url] = true);
    });
  }

  function referencesForNode(node, kind) {
    return window.fetch(node.url).then(function (response) {
      if (!response.ok) throw new Error('Could not load references');
      return response.text();
    }).then(function (html) {
      return localReferences(new window.DOMParser().parseFromString(html, 'text/html'), kind);
    });
  }

  function nodeFromReference(reference, kind) {
    var match = reference.url.match(/\/(problems|solutions)\/([^/]+)\.html/);
    if (!match) return null;
    return {
      id: (match[1] === 'solutions' ? 'solution:' : 'problem:') + match[2],
      title: reference.title,
      type: kind === 'causes' ? 'root cause' : (kind === 'symptoms' ? 'symptom' : (match[1] === 'solutions' ? 'solution' : 'problem')),
      url: reference.url
    };
  }

  function addContextualCausalEdges(trail) {
    var problemNodes = trail.nodes.filter(function (item) { return item.id.indexOf('problem:') === 0; });
    var nodesByUrl = {};
    trail.nodes.forEach(function (item) { nodesByUrl[item.url] = item; });
    return Promise.all(problemNodes.map(function (problemNode) {
      return window.fetch(problemNode.url).then(function (response) {
        if (!response.ok) throw new Error('Could not load causal references');
        return response.text();
      }).then(function (html) {
        var pageDocument = new window.DOMParser().parseFromString(html, 'text/html');
        var symptoms = localReferences(pageDocument, 'symptoms').map(function (reference) {
          var relatedNode = nodesByUrl[reference.url];
          return relatedNode ? { from: relatedNode.id, to: problemNode.id } : null;
        });
        var causes = localReferences(pageDocument, 'causes').map(function (reference) {
          var relatedNode = nodesByUrl[reference.url];
          return relatedNode ? { from: problemNode.id, to: relatedNode.id } : null;
        });
        return symptoms.concat(causes).filter(Boolean);
      }).catch(function () { return []; });
    })).then(function (edgeGroups) {
      var changed = false;
      edgeGroups.flat().forEach(function (candidate) {
        var exists = trail.edges.some(function (edge) {
          return edge.from === candidate.from && edge.to === candidate.to;
        });
        if (!exists && candidate.from !== candidate.to) {
          trail.edges.push({ from: candidate.from, to: candidate.to, label: 'contextual-causes' });
          changed = true;
        }
      });
      if (changed) {
        if (trail.edges.length > maxEdges) trail.edges = trail.edges.slice(-maxEdges);
        saveTrail(trail);
      }
      return changed;
    });
  }

  function nodeX(type, width) {
    var positions = { 'root cause': 0.16, problem: 0.45, symptom: 0.72, solution: 0.88 };
    return Math.round(width * (positions[type] || 0.5));
  }

  function labelLines(title, maximumLength) {
    var lines = [];
    var current = '';
    title.split(/\s+/).forEach(function (word) {
      var candidate = current ? current + ' ' + word : word;
      if (current && candidate.length > maximumLength) {
        lines.push(current);
        current = word;
      } else {
        current = candidate;
      }
    });
    if (current) lines.push(current);
    return lines;
  }

  function render(trail) {
    var container = document.querySelector('[data-analysis-trail-graph]');
    var summary = document.getElementById('analysis-trail-summary');
    if (!container || !summary) return;
    container.innerHTML = '';
    if (!trail.nodes.length) {
      summary.textContent = 'Open a problem or solution to start.';
      container.innerHTML = '<p class="analysis-trail__empty">Your analysis path will appear here.</p>';
      return;
    }

    summary.textContent = trail.nodes.length + (trail.nodes.length === 1 ? ' item in this analysis.' : ' items in this analysis.');
    var typeOrder = { symptom: 0, problem: 1, solution: 2, 'root cause': 3 };
    var displayNodes = trail.nodes.slice().sort(function (first, second) {
      return (typeOrder[first.type] === undefined ? 1 : typeOrder[first.type]) -
        (typeOrder[second.type] === undefined ? 1 : typeOrder[second.type]);
    });
    // Only the focused node is marked. Without an explicit focus no node is
    // active, so a page that carries no article leaves the graph unmarked.
    var currentNodeId = trail.nodes.some(function (item) { return item.id === focusedNodeId; }) ? focusedNodeId : null;
    var width = 1200;
    var height = 900;
    var zoom = Math.max(0.2, Math.min(3, trail.zoom || 2));
    var visibleWidth = width / zoom;
    var visibleHeight = height / zoom;
    var pan = trail.pan || { x: 0, y: 300 };
    var viewBoxX = (width - visibleWidth) / 2 - pan.x;
    var viewBoxY = (height - visibleHeight) / 2 - pan.y;
    var svg = svgElement('svg', { viewBox: viewBoxX + ' ' + viewBoxY + ' ' + visibleWidth + ' ' + visibleHeight, role: 'img', 'aria-label': 'Analysis workbench graph' });
    var defs = svgElement('defs');
    var marker = svgElement('marker', { id: 'analysis-trail-arrow', viewBox: '0 0 10 10', refX: '8', refY: '5', markerWidth: '8', markerHeight: '8', orient: 'auto' });
    marker.appendChild(svgElement('path', { d: 'M 0 0 L 10 5 L 0 10 z', fill: '#cbd5e1' }));
    defs.appendChild(marker);
    svg.appendChild(defs);

    var groupedNodes = { symptom: [], problem: [], solution: [], 'root cause': [] };
    displayNodes.forEach(function (node) {
      (groupedNodes[node.type] || groupedNodes.problem).push(node);
    });
    var rowY = { symptom: 28, problem: 98, solution: 168, 'root cause': 238 };
    // Keep automatically placed nodes close together. Circles have a 20px
    // diameter, so a 40px centre distance leaves roughly 20px of clear space
    // between neighbouring circles while still keeping labels readable.
    var nodeCenterSpacing = 40;
    var positions = {};
    displayNodes.forEach(function (node, index) {
      var group = groupedNodes[node.type] || groupedNodes.problem;
      var groupIndex = group.indexOf(node);
      var rowWidth = Math.max(0, (group.length - 1) * nodeCenterSpacing);
      var defaultPosition = {
        x: Math.max(14, Math.min(width - 14, Math.round((width - rowWidth) / 2 + groupIndex * nodeCenterSpacing))),
        y: rowY[node.type] || rowY.problem
      };
      var savedPosition = trail.positions[node.id];
      positions[node.id] = savedPosition ? {
        x: Math.max(14, Math.min(width - 14, savedPosition.x)),
        y: Math.max(18, Math.min(height - 18, savedPosition.y))
      } : defaultPosition;
    });
    var edgeElements = [];
    var nodeElements = {};
    var nodeMenu;
    var menuHideTimer;
    var menuHideDelay = 500;
    var menuShowTimer;
    var menuShowDelay = 500;
    var activeControlLink;
    // Both controls sit above the circle, side by side: "−" left, "+" right.
    // Keeping the area below the node free lets every label use the same offset.
    var controlOffsetX = 12;
    var controlCircleY = -17;
    var controlTextY = -13;

    function updateEdge(edgeInfo) {
      var from = positions[edgeInfo.edge.from];
      var to = positions[edgeInfo.edge.to];
      if (!from || !to) return;
      var deltaX = to.x - from.x;
      var deltaY = to.y - from.y;
      var distance = Math.max(1, Math.sqrt(deltaX * deltaX + deltaY * deltaY));
      // Start and finish outside the node radius so the arrow remains visible
      // regardless of whether it points up, down, or sideways.
      var gap = 15;
      var startX = from.x + deltaX * gap / distance;
      var startY = from.y + deltaY * gap / distance;
      var endX = to.x - deltaX * gap / distance;
      var endY = to.y - deltaY * gap / distance;
      // Keeping both Bézier control points on the connection vector gives the
      // arrowhead the same angle as the line between the two circles.
      var controlOneX = startX + (endX - startX) / 3;
      var controlOneY = startY + (endY - startY) / 3;
      var controlTwoX = startX + 2 * (endX - startX) / 3;
      var controlTwoY = startY + 2 * (endY - startY) / 3;
      var shape = 'M ' + startX + ' ' + startY + ' C ' + controlOneX + ' ' + controlOneY + ', ' + controlTwoX + ' ' + controlTwoY + ', ' + endX + ' ' + endY;
      edgeInfo.path.setAttribute('d', shape);
      // The visible stroke is too thin to double-click reliably, so an invisible
      // wider path underneath takes the pointer events.
      if (edgeInfo.hit) edgeInfo.hit.setAttribute('d', shape);
    }

    // Double-clicking an edge walks through four states: as drawn, reversed,
    // plain similarity line, and completely deleted.
    function cycleEdge(edge) {
      var isRelated = edge.label === 'related' || /^similar/.test(edge.label || '');
      if (isRelated) {
        trail.edges = trail.edges.filter(function (item) {
          return item !== edge;
        });
        return;
      }
      if (edge.reversed) {
        edge.baseLabel = edge.label;
        edge.label = 'related';
        delete edge.reversed;
        return;
      }
      var from = edge.from;
      edge.from = edge.to;
      edge.to = from;
      edge.reversed = true;
    }

    // A link between two problems is causal, anything involving a solution
    // addresses the problem on the other end.
    function connectNodes(sourceId, targetId) {
      if (sourceId === targetId) return;
      var sourceIsSolution = sourceId.indexOf('solution:') === 0;
      var targetIsSolution = targetId.indexOf('solution:') === 0;
      var edge = targetIsSolution ?
        { from: targetId, to: sourceId, label: 'addresses' } :
        (sourceIsSolution ? { from: sourceId, to: targetId, label: 'addresses' } : { from: sourceId, to: targetId, label: 'causes' });
      var exists = trail.edges.some(function (item) {
        return (item.from === edge.from && item.to === edge.to) || (item.from === edge.to && item.to === edge.from);
      });
      if (exists) return;
      rememberChange(trail);
      trail.edges.push(edge);
      if (trail.edges.length > maxEdges) trail.edges.shift();
      saveTrail(trail);
    }

    function updateNode(nodeId) {
      var elements = nodeElements[nodeId];
      var position = positions[nodeId];
      if (!elements || !position) return;
      if (elements.hoverHit) {
        elements.hoverHit.setAttribute('cx', position.x);
        elements.hoverHit.setAttribute('cy', position.y);
      }
      elements.circle.setAttribute('cx', position.x);
      elements.circle.setAttribute('cy', position.y);
      elements.text.setAttribute('x', position.x);
      elements.text.setAttribute('y', position.y + elements.labelOffset);
      elements.labelLines.forEach(function (line, index) {
        line.setAttribute('x', position.x);
        line.setAttribute('y', position.y + elements.labelOffset + index * 10);
      });
      if (elements.removeIcon) {
        elements.removeHit.setAttribute('cx', position.x - controlOffsetX);
        elements.removeHit.setAttribute('cy', position.y + controlCircleY);
        elements.removeIcon.setAttribute('x', position.x - controlOffsetX);
        elements.removeIcon.setAttribute('y', position.y + controlTextY);
      }
      if (elements.linkNode) {
        elements.linkHit.setAttribute('cx', position.x);
        elements.linkHit.setAttribute('cy', position.y + controlCircleY);
        elements.linkIcon.setAttribute('transform', 'translate(' + position.x + ' ' + (position.y + controlCircleY) + ')');
      }
    }

    function updateGraph() {
      edgeElements.forEach(updateEdge);
      Object.keys(nodeElements).forEach(updateNode);
      Object.keys(nodeElements).forEach(function (nodeId) {
        var addNode = nodeElements[nodeId].addNode;
        var position = positions[nodeId];
        if (!addNode || !position) return;
        addNode.querySelector('circle').setAttribute('cx', position.x + controlOffsetX);
        addNode.querySelector('circle').setAttribute('cy', position.y + controlCircleY);
        addNode.querySelector('text').setAttribute('x', position.x + controlOffsetX);
        addNode.querySelector('text').setAttribute('y', position.y + controlTextY);
      });
    }

    function updateNodeMenuPosition() {
      if (!nodeMenu || !nodeMenu.sourceNode) return;
      var sourceNode = nodeMenu.sourceNode;
      var position = positions[sourceNode.id];
      if (!position) return;
      var left = 0;
      var top = 0;
      if (svg.getScreenCTM && svg.createSVGPoint) {
        var point = svg.createSVGPoint();
        point.x = position.x;
        point.y = position.y;
        var screenPoint = point.matrixTransform(svg.getScreenCTM());
        var containerBounds = container.getBoundingClientRect();
        left = screenPoint.x - containerBounds.left;
        top = screenPoint.y - containerBounds.top + 10;
      } else {
        var svgBounds = svg.getBoundingClientRect();
        var containerBounds = container.getBoundingClientRect();
        left = svgBounds.left - containerBounds.left + (position.x - viewBoxX) * svgBounds.width / visibleWidth;
        top = svgBounds.top - containerBounds.top + (position.y - viewBoxY) * svgBounds.height / visibleHeight + 10;
      }
      nodeMenu.style.left = left + 'px';
      nodeMenu.style.top = top + 'px';
      var scaleFactor = Math.max(0.5, Math.min(3, (trail.zoom || 2) / 2));
      nodeMenu.style.transform = 'scale(' + scaleFactor + ')';
      nodeMenu.style.transformOrigin = 'center center';
    }

    function updateViewBox() {
      svg.setAttribute('viewBox', viewBoxX + ' ' + viewBoxY + ' ' + visibleWidth + ' ' + visibleHeight);
      updateNodeMenuPosition();
    }

    function hideNodeMenu() {
      if (nodeMenu) nodeMenu.remove();
      nodeMenu = null;
      if (activeControlLink) activeControlLink.classList.remove('is-controls-visible');
      activeControlLink = null;
      menuOpenNodeId = null;
    }

    // Opens the reference menu together with the "+", "−" and connect controls.
    // Re-rendering the graph rebuilds every node, so the open menu is restored
    // from menuOpenNodeId instead of disappearing behind an async render.
    function openNodeMenu(nodeId) {
      var elements = nodeElements[nodeId];
      if (!elements) return;
      showNodeMenu(elements.node);
      if (activeControlLink && activeControlLink !== elements.group) activeControlLink.classList.remove('is-controls-visible');
      activeControlLink = elements.group;
      elements.group.classList.add('is-controls-visible');
      menuOpenNodeId = nodeId;
    }

    function loadPageDynamically(url, shouldPushState) {
      if (!url || url === '#') return;
      var mainContent = document.querySelector('.page-main-content');
      if (!mainContent) return;
      // A pending connection is bound to the page it was started on, and its
      // source node would otherwise keep the same blue ring as the active node.
      stopLinking();

      mainContent.style.opacity = '0.5';

      window.fetch(url).then(function (response) {
        if (!response.ok) throw new Error('Could not load page');
        return response.text();
      }).then(function (html) {
        var doc = new window.DOMParser().parseFromString(html, 'text/html');
        var newMain = doc.querySelector('.page-main-content');
        if (!newMain) {
          mainContent.style.opacity = '1';
          return;
        }

        mainContent.innerHTML = newMain.innerHTML;
        mainContent.style.opacity = '1';
        scrollArticleToTop();

        if (doc.title) document.title = doc.title;

        if (shouldPushState !== false && window.history && window.history.pushState) {
          window.history.pushState({ url: url }, doc.title || '', url);
        }

        var article = doc.querySelector('[data-analysis-node]');
        if (article) {
          var newArticleNode = {
            id: article.getAttribute('data-analysis-node-id'),
            title: article.getAttribute('data-analysis-node-title'),
            type: article.getAttribute('data-analysis-node-type'),
            url: url
          };
          node = newArticleNode;
          // The article that is now on screen is the active node, no matter how
          // it was reached.
          setFocusedNodeId(newArticleNode.id);
          addCurrentNode(trail, newArticleNode);
          addPendingEdge(trail, newArticleNode);
          saveTrail(trail);
          render(trail);
          addContextualCausalEdges(trail).then(function (changed) {
            if (changed) render(trail);
          });
        }

        // Focusing a node or a link inside the new article can scroll the page
        // again after the swap, so the offset is reset once more afterwards.
        window.requestAnimationFrame(scrollArticleToTop);
      }).catch(function (error) {
        console.warn('Dynamic page load failed:', error);
        mainContent.style.opacity = '1';
      });
    }

    window.addEventListener('popstate', function (event) {
      if (event.state && event.state.url) {
        loadPageDynamically(event.state.url, false);
      } else {
        loadPageDynamically(window.location.pathname + window.location.search, false);
      }
    });

    function navigateToReference(sourceNode, kind, reference) {
      var relationship = {
        symptoms: { label: 'causes', direction: 'reverse', targetType: 'symptom' },
        causes: { label: 'causes', direction: 'forward', targetType: 'root cause' },
        solutions: { label: 'addresses', direction: 'reverse', targetType: 'solution' },
        'addressed-problems': { label: 'addresses', direction: 'forward', targetType: 'problem' },
        'similar-solutions': { label: 'related', direction: 'forward', targetType: 'solution' },
        'similar-problems': { label: 'related', direction: 'forward', targetType: 'problem' }
      }[kind];
      relationship.from = sourceNode.id;
      window.sessionStorage.setItem(pendingKey, JSON.stringify(relationship));
      loadPageDynamically(reference.url, true);
    }

    function showNodeMenu(sourceNode) {
      window.clearTimeout(menuHideTimer);
      hideNodeMenu();
      if (sourceNode.custom) return;
      var position = positions[sourceNode.id];
      if (!position) return;
      nodeMenu = document.createElement('div');
      nodeMenu.className = 'analysis-trail__node-menu';
      nodeMenu.setAttribute('aria-label', 'References for ' + sourceNode.title);
      nodeMenu.sourceNode = sourceNode;
      updateNodeMenuPosition();

      var menuActions = sourceNode.id.indexOf('solution:') === 0 ?
        [{ kind: 'addressed-problems', label: 'Addressed Problems' }, { kind: 'similar-solutions', label: 'Similar Solutions' }] :
        [{ kind: 'symptoms', label: 'Symptoms' }, { kind: 'causes', label: 'Causes' }, { kind: 'solutions', label: 'Solutions' }, { kind: 'similar-problems', label: 'Similar Problems' }];
      menuActions.forEach(function (menuAction) {
        var kind = menuAction.kind;
        var action = document.createElement('button');
        action.type = 'button';
        action.className = 'analysis-trail__node-menu-action analysis-trail__node-menu-action--' + kind;
        action.textContent = menuAction.label;
        function openActionList(event) {
          if (event) event.stopPropagation();
          var existingList = nodeMenu.querySelector('.analysis-trail__node-menu-list[data-kind="' + kind + '"]');
          if (existingList) return;
          var list = document.createElement('div');
          list.className = 'analysis-trail__node-menu-list';
          list.setAttribute('data-kind', kind);
          var actionBounds = action.getBoundingClientRect();
          var menuBounds = nodeMenu.getBoundingClientRect();
          var actionCenterX = actionBounds.left + actionBounds.width / 2;
          if (actionCenterX < menuBounds.left) {
            // Action button is on left side -> place suggestion list to the left of button (+18px outwards)
            list.style.right = (menuBounds.left - actionBounds.left + 18) + 'px';
            list.style.left = 'auto';
          } else {
            // Action button is on right side -> place suggestion list to the right of button (+18px outwards)
            list.style.left = (actionBounds.right - menuBounds.left + 18) + 'px';
            list.style.right = 'auto';
          }
          list.style.top = (actionBounds.top - menuBounds.top) + 'px';
          list.style.transform = 'none';
          nodeMenu.querySelectorAll('.analysis-trail__node-menu-list').forEach(function (item) { item.remove(); });
          nodeMenu.appendChild(list);
          referencesForNode(sourceNode, kind).then(function (references) {
            list.innerHTML = '';
            if (!references.length) {
              list.textContent = 'No references.';
              return;
            }
            function addReference(reference) {
              var referenceButton = document.createElement('button');
              referenceButton.type = 'button';
              referenceButton.textContent = reference.title;
              referenceButton.addEventListener('click', function () {
                navigateToReference(sourceNode, kind, reference);
              });
              list.appendChild(referenceButton);
            }
            references.forEach(addReference);
            if (references.length) {
              var showAll = document.createElement('button');
              showAll.type = 'button';
              showAll.className = 'analysis-trail__node-menu-show-all';
              showAll.textContent = 'Add all nodes (' + references.length + ')';
              showAll.addEventListener('click', function () {
                showAll.remove();
                references.forEach(function (reference) {
                  var referenceNode = nodeFromReference(reference, kind);
                  if (!referenceNode) return;
                  addCurrentNode(trail, referenceNode);
                  var relation = {
                    symptoms: { from: referenceNode.id, to: sourceNode.id, label: 'causes' },
                    causes: { from: sourceNode.id, to: referenceNode.id, label: 'causes' },
                    solutions: { from: referenceNode.id, to: sourceNode.id, label: 'addresses' },
                    'addressed-problems': { from: sourceNode.id, to: referenceNode.id, label: 'addresses' },
                    'similar-problems': { from: sourceNode.id, to: referenceNode.id, label: 'related' },
                    'similar-solutions': { from: sourceNode.id, to: referenceNode.id, label: 'related' }
                  }[kind];
                  if (relation && relation.from !== relation.to && !trail.edges.some(function (edge) {
                    return edge.from === relation.from && edge.to === relation.to && edge.label === relation.label;
                  })) trail.edges.push(relation);
                });
                saveTrail(trail);
                render(trail);
              });
              list.appendChild(showAll);
            }
          }).catch(function () { list.textContent = 'Could not load references.'; });
        }
        action.addEventListener('click', openActionList);
        nodeMenu.appendChild(action);
      });
      container.appendChild(nodeMenu);
      updateNodeMenuPosition();
    }

    trail.edges.forEach(function (edge) {
      var from = positions[edge.from];
      var to = positions[edge.to];
      if (!from || !to) return;
      // Similar relationships are always plain, light-grey lines. Treat the
      // older descriptive labels as related too so persisted trails cannot
      // accidentally render an arrow for a similarity link.
      var isRelated = edge.label === 'related' || /^similar/.test(edge.label || '');
      var edgeClass = 'analysis-trail__edge analysis-trail__edge--' + (isRelated ? 'related' : edge.label.replace(/\s+/g, '-'));
      var pathAttributes = { class: edgeClass };
      if (!isRelated) pathAttributes['marker-end'] = 'url(#analysis-trail-arrow)';
      var hit = svgElement('path', { class: 'analysis-trail__edge-hit' });
      var hitTitle = svgElement('title');
      hitTitle.textContent = 'Double-click to flip direction, make it a similar link, or remove it';
      hit.appendChild(hitTitle);
      svg.appendChild(hit);
      var path = svgElement('path', pathAttributes);
      svg.appendChild(path);
      var edgeInfo = { edge: edge, path: path, hit: hit };
      hit.addEventListener('dblclick', function (event) {
        event.preventDefault();
        event.stopPropagation();
        rememberChange(trail);
        cycleEdge(edge);
        saveTrail(trail);
        render(trail);
      });
      edgeElements.push(edgeInfo);
      updateEdge(edgeInfo);
    });
    displayNodes.forEach(function (node) {
      var position = positions[node.id];
      // Every node can grow the graph: problems gain causes, symptoms and
      // solutions, solutions gain similar solutions and addressed problems.
      var canAddNode = true;
      // Both controls sit above the node, so every label keeps the same distance.
      var labelOffset = 19;
      // The controls live next to the link, not inside it. Anything nested in an
      // SVG <a> inherits its text decoration, which showed up as stray underline
      // fragments beneath the "+" and "−" glyphs.
      var group = svgElement('g', { class: 'analysis-trail__node-group' });
      var hoverHit = svgElement('circle', { cx: position.x, cy: position.y, r: '36', class: 'analysis-trail__node-hover-hit', fill: 'rgba(0,0,0,0.001)', 'pointer-events': 'all' });
      group.appendChild(hoverHit);
      var link = svgElement('a', { href: node.custom ? '#' : node.url, class: 'analysis-trail__node-link', 'aria-label': node.title });
      var circle = svgElement('circle', { cx: position.x, cy: position.y, r: '10', class: 'analysis-trail__node analysis-trail__node--' + node.type.replace(/\s+/g, '-') + (node.id === currentNodeId ? ' is-current' : '') });
      link.appendChild(circle);
      var text = svgElement('text', { x: position.x, y: position.y + labelOffset, class: 'analysis-trail__node-label', 'text-anchor': 'middle' });
      var lines = labelLines(node.title, 20);
      lines.forEach(function (line, index) {
        var span = svgElement('tspan', { x: position.x, y: position.y + labelOffset + index * 10 });
        span.textContent = line;
        text.appendChild(span);
      });
      link.appendChild(text);
      group.appendChild(link);
      svg.appendChild(group);
      nodeElements[node.id] = { node: node, group: group, link: link, circle: circle, hoverHit: hoverHit, text: text, labelOffset: labelOffset, labelLines: Array.prototype.slice.call(text.querySelectorAll('tspan')) };
      if (canAddNode) {
        var addNode = svgElement('g', { class: 'analysis-trail__add-node', role: 'button', tabindex: '0', 'aria-label': 'Add node to ' + node.title });
        addNode.appendChild(svgElement('circle', { cx: position.x + controlOffsetX, cy: position.y + controlCircleY, r: '6' }));
        var addMark = svgElement('text', { x: position.x + controlOffsetX, y: position.y + controlTextY, 'text-anchor': 'middle' });
        addMark.textContent = '+';
        addNode.appendChild(addMark);
        addNode.addEventListener('click', function (event) { event.preventDefault(); event.stopPropagation(); openAddModal(node, positions[node.id]); });
        group.appendChild(addNode);
        nodeElements[node.id].addNode = addNode;
      }

      // Sits between "−" and "+": start here, then click another node to draw
      // the connection between the two.
      {
        var linkNode = svgElement('g', {
          class: 'analysis-trail__link-node',
          role: 'button',
          tabindex: '0',
          'aria-label': 'Connect ' + node.title + ' to another node'
        });
        var linkHit = svgElement('circle', { cx: position.x, cy: position.y + controlCircleY, r: '6', class: 'analysis-trail__link-hit' });
        // Two small nodes joined by a line, matching what the control creates.
        var linkIcon = svgElement('g', { class: 'analysis-trail__link-icon', transform: 'translate(' + position.x + ' ' + (position.y + controlCircleY) + ')' });
        linkIcon.appendChild(svgElement('line', { x1: '-2', y1: '2', x2: '2', y2: '-2' }));
        linkIcon.appendChild(svgElement('circle', { cx: '-3', cy: '3', r: '1.5' }));
        linkIcon.appendChild(svgElement('circle', { cx: '3', cy: '-3', r: '1.5' }));
        linkNode.appendChild(linkHit);
        linkNode.appendChild(linkIcon);
        var linkTitle = svgElement('title');
        linkTitle.textContent = 'Connect to another node';
        linkNode.appendChild(linkTitle);
        linkNode.addEventListener('click', function (event) {
          event.preventDefault();
          event.stopPropagation();
          if (linkingFrom === node.id) { stopLinking(); return; }
          stopLinking();
          linkingFrom = node.id;
          svg.classList.add('is-linking');
          group.classList.add('is-link-source');
        });
        group.appendChild(linkNode);
        nodeElements[node.id].linkNode = linkNode;
        nodeElements[node.id].linkHit = linkHit;
        nodeElements[node.id].linkIcon = linkIcon;
      }

      // Every node can be removed; any connected arrows are removed as well.
      {
        var remove = svgElement('g', {
          class: 'analysis-trail__remove-node',
          role: 'button',
          tabindex: '0',
          style: 'text-decoration:none',
          'aria-label': 'Remove ' + node.title + ' from the analysis trail'
        });
        var removeHit = svgElement('circle', { cx: position.x - controlOffsetX, cy: position.y + controlCircleY, r: '6', class: 'analysis-trail__remove-hit' });
        var removeIcon = svgElement('text', { x: position.x - controlOffsetX, y: position.y + controlTextY, class: 'analysis-trail__remove-icon', 'text-anchor': 'middle' });
        removeIcon.textContent = '−';
        remove.appendChild(removeHit);
        remove.appendChild(removeIcon);
        var removeTitle = svgElement('title');
        removeTitle.textContent = 'Remove leaf node';
        remove.appendChild(removeTitle);
        var removeNode = function () {
          rememberChange(trail);
          trail.nodes = trail.nodes.filter(function (item) { return item.id !== node.id; });
          trail.edges = trail.edges.filter(function (edge) { return edge.from !== node.id && edge.to !== node.id; });
          delete trail.positions[node.id];
          saveTrail(trail);
          render(trail);
        };
        remove.addEventListener('click', function (event) {
          event.preventDefault();
          event.stopPropagation();
          removeNode();
        });
        remove.addEventListener('keydown', function (event) {
          if (event.key === 'Enter' || event.key === ' ') {
            event.preventDefault();
            removeNode();
          }
        });
        group.appendChild(remove);
        nodeElements[node.id].removeHit = removeHit;
        nodeElements[node.id].removeIcon = removeIcon;
      }
    });
    var draggedNodeId = null;
    var dragged = false;
    var dragStart;
    var suppressClick = false;
    var panning = false;
    var panStart;
    var panChanged = false;

    function svgPoint(event) {
      var bounds = svg.getBoundingClientRect();
      return {
        x: viewBoxX + (event.clientX - bounds.left) * visibleWidth / bounds.width,
        y: viewBoxY + (event.clientY - bounds.top) * visibleHeight / bounds.height
      };
    }

    // A re-render rebuilds every node, so a connection started before it has to
    // be marked again — or dropped if its source node is gone.
    if (linkingFrom && !nodeElements[linkingFrom]) stopLinking();
    if (linkingFrom) {
      svg.classList.add('is-linking');
      nodeElements[linkingFrom].group.classList.add('is-link-source');
    }

    // Exactly one node carries the focus ring, so switching focus always clears
    // the ring and the controls of every other node.
    function markActiveNode(nodeId) {
      Object.keys(nodeElements).forEach(function (id) {
        var elements = nodeElements[id];
        if (id === nodeId) elements.circle.classList.add('is-current');
        else elements.circle.classList.remove('is-current');
        if (id !== nodeId) elements.group.classList.remove('is-controls-visible');
      });
    }

    // 2-Step Click Model:
    // 1st click on an unfocused node: focuses it and loads its article.
    // 2nd click on the focused node: toggles the reference menus and controls.
    function activateNode(nodeId) {
      var nodeElement = nodeElements[nodeId];
      if (!nodeElement) return;

      if (currentNodeId === nodeId) {
        if (menuOpenNodeId === nodeId) hideNodeMenu();
        else openNodeMenu(nodeId);
        return;
      }

      hideNodeMenu();
      currentNodeId = nodeId;
      setFocusedNodeId(nodeId);
      markActiveNode(nodeId);
      // A focused SVG link makes the browser scroll it into view, which would
      // fight the scroll reset of the freshly loaded article.
      if (document.activeElement && document.activeElement.blur) document.activeElement.blur();
      if (!nodeElement.node.custom && nodeElement.node.url && nodeElement.node.url !== '#') {
        loadPageDynamically(nodeElement.node.url, true);
      }
    }

    Object.keys(nodeElements).forEach(function (nodeId) {
      var nodeElement = nodeElements[nodeId];

      nodeElement.link.addEventListener('pointerdown', function (event) {
        if (event.button !== 0) return;
        if (spacePressed) return;
        event.stopPropagation();
        draggedNodeId = nodeId;
        dragged = false;
        dragStart = { clientX: event.clientX, clientY: event.clientY };
        svg.setPointerCapture(event.pointerId);
      });

      // Pointer interaction is handled once, on pointer up, so a click can never
      // load the same article twice. Only keyboard activation, which produces a
      // click without a pointer sequence, is handled here.
      nodeElement.link.addEventListener('click', function (event) {
        event.preventDefault();
        if (suppressClick) {
          suppressClick = false;
          return;
        }
        if (event.detail !== 0) return;
        if (linkingFrom) {
          if (linkingFrom !== nodeId) {
            var sourceId = linkingFrom;
            stopLinking();
            connectNodes(sourceId, nodeId);
            render(trail);
          }
          return;
        }
        activateNode(nodeId);
      });
      nodeElement.link.addEventListener('dblclick', function (event) {
        event.preventDefault();
      });
    });

    svg.addEventListener('pointerdown', function (event) {
      var isMiddleMouse = event.button === 1;
      var isSpacePan = event.button === 0 && spacePressed;
      if (!isMiddleMouse && !isSpacePan && event.target !== svg) return;
      if (!isMiddleMouse && !isSpacePan && event.button !== 0) return;
      event.preventDefault();
      hideNodeMenu();
      stopLinking();
      panning = true;
      panChanged = false;
      var bounds = svg.getBoundingClientRect();
      panStart = {
        clientX: event.clientX,
        clientY: event.clientY,
        scaleX: visibleWidth / bounds.width,
        scaleY: visibleHeight / bounds.height,
        pan: snapshot(trail.pan || { x: 0, y: 300 })
      };
      svg.setPointerCapture(event.pointerId);
      svg.classList.add('is-panning');
    });

    svg.addEventListener('pointermove', function (event) {
      var point = svgPoint(event);
      if (draggedNodeId) {
        if (!dragged) {
          var movementX = event.clientX - dragStart.clientX;
          var movementY = event.clientY - dragStart.clientY;
          if (Math.sqrt(movementX * movementX + movementY * movementY) < 4) return;
          dragged = true;
        }
        positions[draggedNodeId] = {
          x: Math.max(14, Math.min(width - 14, point.x)),
          y: Math.max(18, Math.min(height - 18, point.y))
        };
        updateGraph();
      } else if (panning) {
        if (!panChanged) rememberChange(trail);
        panChanged = true;
        trail.pan = {
          x: panStart.pan.x + (event.clientX - panStart.clientX) * panStart.scaleX,
          y: panStart.pan.y + (event.clientY - panStart.clientY) * panStart.scaleY
        };
        viewBoxX = (width - visibleWidth) / 2 - trail.pan.x;
        viewBoxY = (height - visibleHeight) / 2 - trail.pan.y;
        updateViewBox();
      }
    });
    svg.addEventListener('pointerup', function (event) {
      var releasedNodeId = draggedNodeId;
      if (draggedNodeId && dragged) {
        rememberChange(trail);
        trail.positions[draggedNodeId] = positions[draggedNodeId];
        saveTrail(trail);
        suppressClick = true;
      }
      if (releasedNodeId && !dragged) {
        if (linkingFrom) {
          if (linkingFrom !== releasedNodeId) {
            var sourceId = linkingFrom;
            stopLinking();
            connectNodes(sourceId, releasedNodeId);
            render(trail);
          } else {
            stopLinking();
          }
        } else {
          activateNode(releasedNodeId);
        }
      }
      if (svg.hasPointerCapture(event.pointerId)) svg.releasePointerCapture(event.pointerId);
      draggedNodeId = null;
      dragStart = null;
      if (panning && panChanged) {
        saveTrail(trail);
        suppressClick = true;
      }
      panning = false;
      svg.classList.remove('is-panning');
    });
    container.appendChild(svg);
    // The menu is positioned from the live geometry, so it can only be restored
    // once the new graph sits in the document.
    if (menuOpenNodeId && nodeElements[menuOpenNodeId]) openNodeMenu(menuOpenNodeId);
    else menuOpenNodeId = null;
  }

  document.addEventListener('DOMContentLoaded', function () {
    updateMenuCount();
    var speedNav = document.querySelector('[data-analysis-trail-speed-nav]');
    if (speedNav) {
      var savedSpeedNav = window.sessionStorage.getItem(speedNavKey);
      if (savedSpeedNav !== null) speedNav.checked = savedSpeedNav === 'true';
      speedNav.addEventListener('change', function () {
        window.sessionStorage.setItem(speedNavKey, String(speedNav.checked));
      });
    }

    // Number keys provide quick section navigation on problem and solution
    // detail pages: 1 = first heading, 2 = second heading, and so on.
    document.addEventListener('keydown', function (event) {
      if (!/^[1-9]$/.test(event.key) || event.ctrlKey || event.metaKey || event.altKey || event.shiftKey) return;
      if (speedNav && !speedNav.checked) return;
      if (/^(INPUT|TEXTAREA|SELECT|BUTTON)$/.test(event.target.tagName) || event.target.isContentEditable) return;
      var headings = Array.prototype.slice.call(document.querySelectorAll('.page-main-content h1, .page-main-content h2, .page-main-content h3'));
      var heading = headings[Number(event.key) - 1];
      if (!heading) return;
      event.preventDefault();
      heading.setAttribute('tabindex', '-1');
      heading.scrollIntoView({ behavior: 'smooth', block: 'start' });
      heading.focus({ preventScroll: true });
    });

    document.addEventListener('keydown', function (event) {
      if (event.key === 'Escape') stopLinking();
      if (event.code === 'Space' && !/^(INPUT|TEXTAREA|SELECT)$/.test(event.target.tagName)) {
        spacePressed = true;
        var graph = document.querySelector('[data-analysis-trail-graph] svg');
        if (graph) graph.classList.add('is-space-panning');
      }
    });
    document.addEventListener('keyup', function (event) {
      if (event.code === 'Space') {
        spacePressed = false;
        var graph = document.querySelector('[data-analysis-trail-graph] svg');
        if (graph) graph.classList.remove('is-space-panning');
      }
    });
    var trail = getTrail();
    var node = currentNode();
    // Entering a page defines the active node. Any node that was active in an
    // earlier session or on the previous page is deactivated here, so the graph
    // never shows more than one focus ring.
    setFocusedNodeId(node ? node.id : null);
    var addModal = document.querySelector('[data-analysis-trail-add-modal]');
    var addModalText = document.querySelector('[data-analysis-trail-add-modal-text]');
    if (addModal && addModal.parentElement !== document.body) document.body.appendChild(addModal);
    var addSearchResults = document.querySelector('[data-analysis-trail-add-search-results]');
    var addSelection = document.querySelector('[data-analysis-trail-add-selection]');
    var catalog = { problems: [], solutions: [] };
    try { catalog = JSON.parse(document.querySelector('[data-analysis-trail-catalog]').textContent); } catch (error) { catalog = { problems: [], solutions: [] }; }

    function closeAddModal() {
      addModal.hidden = true;
      pendingAdd = null;
      addSearchResults.innerHTML = '';
      addSearchResults.hidden = true;
    }

    function updateAddSearch() {
      if (!pendingAdd) return;
      pendingAdd.selected = null;
      addSelection.textContent = 'Custom node will be created.';
      addSearchResults.innerHTML = '';
      var query = addModalText.value.trim().toLowerCase();
      if (!query) { addSearchResults.hidden = true; return; }
      var pool = addTypeUsesSolutionCatalog(currentAddType) ? catalog.solutions : catalog.problems;
      var matches = pool.filter(function (item) {
        return item.id !== pendingAdd.sourceNode.id && item.title.toLowerCase().indexOf(query) !== -1;
      }).slice(0, 8);
      matches.forEach(function (item) {
        var result = document.createElement('button');
        result.type = 'button';
        result.setAttribute('role', 'option');
        result.textContent = item.title;
        result.addEventListener('click', function () {
          pendingAdd.selected = item;
          addModalText.value = item.title;
          addSelection.textContent = 'Existing ' + (addTypeUsesSolutionCatalog(currentAddType) ? 'solution' : 'problem') + ' selected.';
          addSearchResults.hidden = true;
        });
        addSearchResults.appendChild(result);
      });
      addSearchResults.hidden = matches.length === 0;
    }

    var typeButtons = document.querySelectorAll('.analysis-trail__type-btn');
    typeButtons.forEach(function (btn) {
      btn.addEventListener('click', function () {
        currentAddType = btn.getAttribute('data-type');
        updateTypeButtonsDOM();
        updateAddSearch();
      });
    });

    addModalText.addEventListener('input', updateAddSearch);
    document.querySelector('[data-analysis-trail-add-modal-cancel]').addEventListener('click', closeAddModal);
    document.querySelector('[data-analysis-trail-add-modal-submit]').addEventListener('click', function () {
      if (!pendingAdd || !addModalText.value.trim()) return;
      var sourceNode = pendingAdd.sourceNode;
      var kind = currentAddType;
      var selected = pendingAdd.selected;
      if (!selected) {
        var exactPool = addTypeUsesSolutionCatalog(kind) ? catalog.solutions : catalog.problems;
        selected = exactPool.filter(function (item) {
          return item.id !== sourceNode.id && item.title.toLowerCase() === addModalText.value.trim().toLowerCase();
        })[0] || null;
      }
      var targetId = selected ? selected.id : 'custom:' + Date.now() + '-' + Math.random().toString(36).slice(2, 8);
      var nodeType = 'problem';
      if (kind === 'cause') nodeType = 'root cause';
      else if (kind === 'symptom') nodeType = 'symptom';
      else if (addTypeUsesSolutionCatalog(kind)) nodeType = 'solution';
      var targetNode = {
        id: targetId,
        title: selected ? selected.title : addModalText.value.trim(),
        type: nodeType,
        custom: !selected,
        url: selected ? selected.url : '#'
      };
      rememberChange(trail);
      addCurrentNode(trail, targetNode);
      var sourcePosition = pendingAdd.position;
      trail.positions[targetId] = { x: Math.min(1186, sourcePosition.x + 190), y: sourcePosition.y };
      // A similar problem or solution gets the plain, undirected similarity
      // line; causes point away from the source, symptoms and solutions point
      // back to it, and a solution always points at the problem it addresses.
      var edge = (kind === 'similar' || kind === 'similar-solution') ? { from: sourceNode.id, to: targetId, label: 'related' } :
        (kind === 'cause' ? { from: sourceNode.id, to: targetId, label: 'causes' } :
          (kind === 'addressed-problem' ? { from: sourceNode.id, to: targetId, label: 'addresses' } :
            { from: targetId, to: sourceNode.id, label: kind === 'solution' ? 'addresses' : 'causes' }));
      if (!trail.edges.some(function (item) { return item.from === edge.from && item.to === edge.to && item.label === edge.label; })) trail.edges.push(edge);
      if (trail.edges.length > maxEdges) trail.edges.shift();
      saveTrail(trail);
      closeAddModal();
      render(trail);
    });
    addModalText.addEventListener('keydown', function (event) {
      if (event.key === 'Enter') document.querySelector('[data-analysis-trail-add-modal-submit]').click();
      if (event.key === 'Escape') closeAddModal();
    });

    // Quick search reuses the same catalog as the add-node search, but it drops
    // the hit straight onto the canvas instead of attaching it to a source node.
    var quickSearch = document.querySelector('[data-analysis-trail-quick-search]');
    var quickFilter = document.querySelector('[data-analysis-trail-quick-filter]');
    var quickResults = document.querySelector('[data-analysis-trail-quick-results]');

    function quickPool() {
      var kind = quickFilter ? quickFilter.value : 'all';
      var problems = kind === 'solution' ? [] : catalog.problems.map(function (item) { return { item: item, type: 'problem' }; });
      var solutions = kind === 'problem' ? [] : catalog.solutions.map(function (item) { return { item: item, type: 'solution' }; });
      return problems.concat(solutions);
    }

    function freePosition() {
      // Drop new nodes into the first free slot of a coarse grid so repeated
      // searches do not stack every hit on the same spot.
      var used = Object.keys(trail.positions).map(function (id) { return trail.positions[id]; });
      for (var row = 0; row < 20; row++) {
        for (var column = 0; column < 6; column++) {
          var candidate = { x: 120 + column * 190, y: 60 + row * 90 };
          var taken = used.some(function (position) {
            return Math.abs(position.x - candidate.x) < 90 && Math.abs(position.y - candidate.y) < 50;
          });
          if (!taken) return candidate;
        }
      }
      return { x: 120, y: 60 };
    }

    function addQuickNode(entry) {
      var existing = trail.nodes.some(function (item) { return item.id === entry.item.id; });
      rememberChange(trail);
      addCurrentNode(trail, {
        id: entry.item.id,
        title: entry.item.title,
        type: entry.type,
        custom: false,
        url: entry.item.url
      });
      if (!existing) trail.positions[entry.item.id] = freePosition();
      saveTrail(trail);
      render(trail);
    }

    function updateQuickSearch() {
      if (!quickResults) return;
      quickResults.innerHTML = '';
      var query = quickSearch.value.trim().toLowerCase();
      if (!query) { quickResults.hidden = true; return; }
      var matches = quickPool().filter(function (entry) {
        return entry.item.title.toLowerCase().indexOf(query) !== -1;
      }).slice(0, 8);
      matches.forEach(function (entry) {
        var result = document.createElement('button');
        result.type = 'button';
        result.setAttribute('role', 'option');
        var badge = document.createElement('span');
        badge.className = 'analysis-trail__quick-badge analysis-trail__quick-badge--' + entry.type;
        badge.textContent = entry.type === 'solution' ? 'Solution' : 'Problem';
        result.appendChild(badge);
        result.appendChild(document.createTextNode(entry.item.title));
        result.addEventListener('click', function () {
          quickSearch.value = '';
          quickResults.hidden = true;
          addQuickNode(entry);
        });
        quickResults.appendChild(result);
      });
      quickResults.hidden = matches.length === 0;
    }

    if (quickSearch && quickResults) {
      quickSearch.addEventListener('input', updateQuickSearch);
      if (quickFilter) quickFilter.addEventListener('change', updateQuickSearch);
      quickSearch.addEventListener('keydown', function (event) {
        if (event.key === 'Escape') { quickSearch.value = ''; quickResults.hidden = true; }
        if (event.key === 'Enter') {
          var first = quickResults.querySelector('button');
          if (first) first.click();
        }
      });
      document.addEventListener('click', function (event) {
        if (!quickResults.contains(event.target) && event.target !== quickSearch) quickResults.hidden = true;
      });
    }
    addCurrentNode(trail, node);
    addPendingEdge(trail, node);
    saveTrail(trail);
    render(trail);
    addContextualCausalEdges(trail).then(function (changed) {
      if (changed) render(trail);
    });

    var trailLayout = document.querySelector('.page-with-analysis-trail');
    var modeButton = document.querySelector('[data-analysis-trail-mode]');
    var openButton = document.querySelector('[data-analysis-trail-open]');
    function setExpanded(isExpanded) {
      if (!trailLayout) return;
      trailLayout.classList.toggle('is-analysis-trail-expanded', isExpanded);
      document.body.classList.toggle('analysis-trail-expanded', isExpanded);
      if (modeButton) modeButton.setAttribute('aria-pressed', String(isExpanded));
      window.sessionStorage.setItem(expandedKey, String(isExpanded));
      render(trail);
    }
    if (window.sessionStorage.getItem(expandedKey) === 'true') setExpanded(true);
    if (modeButton) modeButton.addEventListener('click', function () {
      setExpanded(!trailLayout.classList.contains('is-analysis-trail-expanded'));
    });
    if (openButton) openButton.addEventListener('click', function () { setExpanded(true); });

    var splitKey = 'problemrider-analysis-trail-split-v1';
    var savedSplit = window.sessionStorage.getItem(splitKey);
    if (savedSplit) {
      document.documentElement.style.setProperty('--analysis-trail-split', savedSplit + '%');
    }

    var resizer = document.querySelector('[data-analysis-trail-resizer]');
    if (resizer) {
      var isResizing = false;

      resizer.addEventListener('pointerdown', function (event) {
        if (event.button !== 0) return;
        event.preventDefault();
        isResizing = true;
        resizer.classList.add('is-dragging');
        resizer.setPointerCapture(event.pointerId);
      });

      resizer.addEventListener('pointermove', function (event) {
        if (!isResizing) return;
        var windowWidth = window.innerWidth;
        if (windowWidth <= 700) return;
        var percent = (event.clientX / windowWidth) * 100;
        percent = Math.max(20, Math.min(80, percent));
        document.documentElement.style.setProperty('--analysis-trail-split', percent + '%');
        window.sessionStorage.setItem(splitKey, percent.toFixed(2));
      });

      resizer.addEventListener('pointerup', function (event) {
        if (isResizing) {
          isResizing = false;
          resizer.classList.remove('is-dragging');
          if (resizer.hasPointerCapture(event.pointerId)) {
            resizer.releasePointerCapture(event.pointerId);
          }
          render(trail);
        }
      });

      resizer.addEventListener('pointercancel', function (event) {
        if (isResizing) {
          isResizing = false;
          resizer.classList.remove('is-dragging');
        }
      });
    }

    document.addEventListener('click', function (event) {
      var link = event.target.closest('a[href]');
      if (!link || link.target === '_blank' || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
      var target;
      try { target = new URL(link.href, window.location.href); } catch (error) { return; }
      if (target.origin !== window.location.origin) return;
      if (!/\.html$/.test(target.pathname) && !/\/(problems|solutions)\//.test(target.pathname)) return;
      event.preventDefault();
      if (node) {
        var edge = edgeForLink(link);
        if (edge) {
          edge.from = node.id;
          window.sessionStorage.setItem(pendingKey, JSON.stringify(edge));
        }
      }
      loadPageDynamically(target.pathname + target.search, true);
    });

    var reset = document.querySelector('[data-analysis-trail-reset]');
    if (reset) reset.addEventListener('click', function () {
      rememberChange(trail);
      window.sessionStorage.removeItem(storageKey);
      window.sessionStorage.removeItem(pendingKey);
      var freshTrail = { nodes: [], edges: [], positions: {}, pan: { x: 0, y: 300 }, zoom: 2 };
      if (node) addCurrentNode(freshTrail, node);
      trail = freshTrail;
      saveTrail(freshTrail);
      render(freshTrail);
    });

    function changeZoom(change) {
      rememberChange(trail);
      trail.zoom = Math.max(0.2, Math.min(3, (trail.zoom || 2) + change));
      saveTrail(trail);
      render(trail);
    }
    var zoomIn = document.querySelector('[data-analysis-trail-zoom-in]');
    var zoomOut = document.querySelector('[data-analysis-trail-zoom-out]');
    if (zoomIn) zoomIn.addEventListener('click', function () { changeZoom(0.2); });
    if (zoomOut) zoomOut.addEventListener('click', function () { changeZoom(-0.2); });

    function graphSvgForRasterExport() {
      var svg = document.querySelector('[data-analysis-trail-graph] svg');
      if (!svg) return null;
      var exportSvg = svg.cloneNode(true);
      exportSvg.setAttribute('xmlns', 'http://www.w3.org/2000/svg');
      exportSvg.setAttribute('xmlns:xlink', 'http://www.w3.org/1999/xlink');
      exportSvg.setAttribute('style', 'background:#fff;font-family:Arial,sans-serif');
      exportSvg.querySelectorAll('.analysis-trail__edge').forEach(function (edge) {
        edge.setAttribute('fill', 'none');
        edge.setAttribute('stroke', edge.classList.contains('analysis-trail__edge--related') ? '#f1f5f9' : (edge.classList.contains('analysis-trail__edge--contextual-causes') ? '#e2e8f0' : '#cbd5e1'));
        edge.setAttribute('stroke-width', edge.classList.contains('analysis-trail__edge--related') ? '1' : '1.6');
        if (edge.classList.contains('analysis-trail__edge--addresses')) edge.setAttribute('stroke-dasharray', '5 3');
      });
      exportSvg.querySelectorAll('.analysis-trail__node').forEach(function (nodeElement) {
        var solution = nodeElement.classList.contains('analysis-trail__node--solution');
        nodeElement.setAttribute('fill', solution ? '#007acc' : '#111');
        nodeElement.setAttribute('stroke', nodeElement.classList.contains('is-current') ? (solution ? '#111' : '#007acc') : (solution ? 'transparent' : '#fff'));
        nodeElement.setAttribute('stroke-width', '2');
      });
      exportSvg.querySelectorAll('.analysis-trail__node-label').forEach(function (label) {
        label.setAttribute('fill', '#555');
        label.removeAttribute('stroke');
        label.removeAttribute('stroke-width');
        label.removeAttribute('paint-order');
        label.setAttribute('font-family', 'Arial, sans-serif');
        label.setAttribute('font-size', '8');
      });
      exportSvg.querySelectorAll('.analysis-trail__remove-node,.analysis-trail__add-node').forEach(function (control) { control.remove(); });
      var viewBox = svg.viewBox.baseVal;
      exportSvg.setAttribute('width', viewBox.width);
      exportSvg.setAttribute('height', viewBox.height);
      return { source: new XMLSerializer().serializeToString(exportSvg), width: viewBox.width, height: viewBox.height };
    }

    function pngWithDpi(blob, dpi) {
      return blob.arrayBuffer().then(function (buffer) {
        var source = new Uint8Array(buffer);
        var pixelsPerMeter = Math.round(dpi / 0.0254);
        var type = new Uint8Array([112, 72, 89, 115]);
        var data = new Uint8Array(9);
        var dataView = new DataView(data.buffer);
        dataView.setUint32(0, pixelsPerMeter);
        dataView.setUint32(4, pixelsPerMeter);
        data[8] = 1;
        var crcInput = new Uint8Array(type.length + data.length);
        crcInput.set(type); crcInput.set(data, type.length);
        var crc = 0xffffffff;
        for (var index = 0; index < crcInput.length; index++) {
          crc ^= crcInput[index];
          for (var bit = 0; bit < 8; bit++) crc = (crc >>> 1) ^ ((crc & 1) ? 0xedb88320 : 0);
        }
        crc = (crc ^ 0xffffffff) >>> 0;
        var chunk = new Uint8Array(21);
        var chunkView = new DataView(chunk.buffer);
        chunkView.setUint32(0, 9);
        chunk.set(type, 4);
        chunk.set(data, 8);
        chunkView.setUint32(17, crc);
        var result = new Uint8Array(source.length + chunk.length);
        result.set(source.slice(0, 33), 0);
        result.set(chunk, 33);
        result.set(source.slice(33), 54);
        return new Blob([result], { type: 'image/png' });
      });
    }

    var pngExportButton = document.querySelector('[data-analysis-trail-png-export]');
    if (pngExportButton) pngExportButton.addEventListener('click', function () {
      var exported = graphSvgForRasterExport();
      if (!exported) return;
      var scale = 300 / 96;
      var canvas = document.createElement('canvas');
      canvas.width = Math.max(1, Math.round(exported.width * scale));
      canvas.height = Math.max(1, Math.round(exported.height * scale));
      var context = canvas.getContext('2d');
      context.fillStyle = '#fff';
      context.fillRect(0, 0, canvas.width, canvas.height);
      var image = new Image();
      var svgUrl = URL.createObjectURL(new Blob([exported.source], { type: 'image/svg+xml;charset=utf-8' }));
      image.onload = function () {
        context.drawImage(image, 0, 0, canvas.width, canvas.height);
        URL.revokeObjectURL(svgUrl);
        canvas.toBlob(function (blob) {
          if (!blob) return;
          pngWithDpi(blob, 300).then(function (pngBlob) {
            var link = document.createElement('a');
            link.href = URL.createObjectURL(pngBlob);
            var timestamp = new Date().toISOString().replace(/[-:]/g, '').replace(/\.\d{3}Z$/, 'Z');
            link.download = 'analysis-workbench-300dpi-' + timestamp + '.png';
            document.body.appendChild(link);
            link.click();
            link.remove();
            URL.revokeObjectURL(link.href);
          });
        }, 'image/png');
      };
      image.onerror = function () { URL.revokeObjectURL(svgUrl); window.alert('Could not create the PNG export.'); };
      image.src = svgUrl;
    });

    var exportButton = document.querySelector('[data-analysis-trail-export]');
    if (exportButton) exportButton.addEventListener('click', function () {
      var svg = document.querySelector('[data-analysis-trail-graph] svg');
      if (!svg) return;
      var exportSvg = svg.cloneNode(true);
      exportSvg.setAttribute('xmlns', 'http://www.w3.org/2000/svg');
      exportSvg.setAttribute('xmlns:xlink', 'http://www.w3.org/1999/xlink');
      exportSvg.setAttribute('style', 'background:#fff;font-family:Arial,sans-serif');
      exportSvg.querySelectorAll('.analysis-trail__edge').forEach(function (edge) {
        edge.setAttribute('fill', 'none');
        edge.setAttribute('stroke', edge.classList.contains('analysis-trail__edge--related') ? '#f1f5f9' : (edge.classList.contains('analysis-trail__edge--contextual-causes') ? '#e2e8f0' : '#cbd5e1'));
        edge.setAttribute('stroke-width', edge.classList.contains('analysis-trail__edge--related') ? '1' : '1.6');
        if (edge.classList.contains('analysis-trail__edge--addresses')) edge.setAttribute('stroke-dasharray', '5 3');
        if (!edge.classList.contains('analysis-trail__edge--related')) edge.setAttribute('marker-end', 'url(#analysis-trail-arrow)');
      });
      exportSvg.querySelectorAll('.analysis-trail__node').forEach(function (nodeElement) {
        nodeElement.setAttribute('fill', nodeElement.classList.contains('analysis-trail__node--solution') ? '#007acc' : '#111');
        nodeElement.setAttribute('stroke', nodeElement.classList.contains('is-current') ? (nodeElement.classList.contains('analysis-trail__node--solution') ? '#111' : '#007acc') : (nodeElement.classList.contains('analysis-trail__node--solution') ? 'transparent' : '#fff'));
        nodeElement.setAttribute('stroke-width', '2');
      });
      exportSvg.querySelectorAll('.analysis-trail__node-label').forEach(function (label) {
        label.setAttribute('fill', '#555');
        label.removeAttribute('stroke');
        label.removeAttribute('stroke-width');
        label.removeAttribute('paint-order');
        label.setAttribute('font-family', 'Arial, sans-serif');
        label.setAttribute('font-size', '8');
      });
      var exportStyle = document.createElementNS(namespace, 'style');
      exportStyle.textContent = [
        '.analysis-trail__edge{fill:none;stroke:#cbd5e1;stroke-width:1.6}',
        '.analysis-trail__edge--contextual-causes{stroke:#e2e8f0;stroke-width:1.2}',
        '.analysis-trail__edge--related{stroke:#f1f5f9;stroke-width:1}',
        '.analysis-trail__edge--addresses{stroke-dasharray:5 3}',
        '.analysis-trail__node{fill:#111;stroke:#fff;stroke-width:2}',
        '.analysis-trail__node--solution{fill:#007acc;stroke:transparent}',
        '.analysis-trail__node--solution.is-current{stroke:#111}',
        '.analysis-trail__node-label{fill:#555;font-size:8px}',
        '.analysis-trail__remove-node{display:none}'
      ].join('');
      exportSvg.insertBefore(exportStyle, exportSvg.firstChild);
      var source = new XMLSerializer().serializeToString(exportSvg);
      var blob = new Blob([source], { type: 'image/svg+xml;charset=utf-8' });
      var url = URL.createObjectURL(blob);
      var link = document.createElement('a');
      link.href = url;
      var timestamp = new Date().toISOString().replace(/[-:]/g, '').replace(/\.\d{3}Z$/, 'Z');
      link.download = 'analysis-workbench-' + timestamp + '.svg';
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
    });

    var drawioExportButton = document.querySelector('[data-analysis-trail-drawio-export]');
    if (drawioExportButton) drawioExportButton.addEventListener('click', function () {
      function xmlEscape(value) {
        return String(value == null ? '' : value)
          .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
          .replace(/"/g, '&quot;').replace(/'/g, '&apos;');
      }

      var cells = [
        '<mxCell id="0"/>',
        '<mxCell id="1" parent="0"/>'
      ];
      trail.nodes.forEach(function (node) {
        var position = trail.positions[node.id] || { x: 0, y: 0 };
        var isSolution = node.type === 'solution';
        var fill = isSolution ? '#c9daf8' : '#f4cccc';
        var stroke = isSolution ? '#1155cc' : '#cc0000';
        var cellId = xmlEscape(node.id);
        cells.push('<mxCell id="' + cellId + '" value="' + xmlEscape(node.title) + '" style="rounded=1;whiteSpace=wrap;html=1;fillColor=' + fill + ';strokeColor=' + stroke + ';" vertex="1" parent="1"><mxGeometry x="' + Number(position.x || 0) + '" y="' + Number(position.y || 0) + '" width="180" height="60" as="geometry"/></mxCell>');
      });
      trail.edges.forEach(function (edge, index) {
        var style = 'endArrow=block;endFill=1;strokeColor=#adb5bd;';
        if (edge.label === 'related') style += 'dashed=1;';
        cells.push('<mxCell id="edge-' + index + '" value="' + xmlEscape(edge.label || '') + '" style="' + style + '" edge="1" parent="1" source="' + xmlEscape(edge.from) + '" target="' + xmlEscape(edge.to) + '"><mxGeometry relative="1" as="geometry"/></mxCell>');
      });

      var source = '<?xml version="1.0" encoding="UTF-8"?><mxfile host="app.diagrams.net"><diagram name="Analysis Workbench"><mxGraphModel><root>' + cells.join('') + '</root></mxGraphModel></diagram></mxfile>';
      var blob = new Blob([source], { type: 'application/xml;charset=utf-8' });
      var link = document.createElement('a');
      link.href = URL.createObjectURL(blob);
      var timestamp = new Date().toISOString().replace(/[-:]/g, '').replace(/\.\d{3}Z$/, 'Z');
      link.download = 'analysis-workbench-drawio-' + timestamp + '.drawio';
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(link.href);
    });

    var excalidrawExportButton = document.querySelector('[data-analysis-trail-excalidraw-export]');
    if (excalidrawExportButton) excalidrawExportButton.addEventListener('click', function () {
      function id(prefix, index) { return prefix + '-' + index + '-' + Date.now(); }
      function colorFor(node) { return node.type === 'solution' ? { background: '#c9daf8', stroke: '#1155cc' } : { background: '#f4cccc', stroke: '#cc0000' }; }
      function positionFor(node) {
        var position = trail.positions[node.id] || { x: 0, y: 0 };
        var spread = 2.5;
        return { x: 600 + (position.x - 600) * spread, y: 450 + (position.y - 450) * spread };
      }
      var elements = [];
      var elementIds = {};
      var boxElements = {};
      trail.nodes.forEach(function (node, index) {
        var position = positionFor(node);
        var colors = colorFor(node);
        var boxId = id('box', index);
        var textId = id('text', index);
        elementIds[node.id] = boxId;
        var box = { type: 'rectangle', version: 1, versionNonce: index + 1, isDeleted: false, id: boxId, fillStyle: 'solid', strokeWidth: 2, strokeStyle: 'solid', roughness: 0, opacity: 100, angle: 0, x: position.x - 90, y: position.y - 30, strokeColor: colors.stroke, backgroundColor: colors.background, width: 180, height: 60, seed: index + 1, groupIds: [], roundness: { type: 3 }, boundElements: [{ id: textId, type: 'text' }], updated: Date.now(), link: null, locked: false };
        boxElements[node.id] = box;
        elements.push(box);
        elements.push({ type: 'text', version: 1, versionNonce: index + 1001, isDeleted: false, id: textId, text: node.title, originalText: node.title, autoResize: false, lineHeight: 1.25, baseline: 16, fontSize: 14, fontFamily: 1, textAlign: 'center', verticalAlign: 'middle', containerId: boxId, strokeColor: '#1e1e1e', backgroundColor: 'transparent', fillStyle: 'solid', strokeWidth: 1, strokeStyle: 'solid', roughness: 0, opacity: 100, angle: 0, x: position.x - 80, y: position.y - 10, width: 160, height: 20, seed: index + 2001, groupIds: [], boundElements: [], updated: Date.now(), link: node.custom ? null : node.url, locked: false });
      });
      trail.edges.forEach(function (edge, index) {
        var from = positionFor(trail.nodes.filter(function (node) { return node.id === edge.from; })[0] || {});
        var to = positionFor(trail.nodes.filter(function (node) { return node.id === edge.to; })[0] || {});
        var fromId = elementIds[edge.from];
        var toId = elementIds[edge.to];
        if (!fromId || !toId) return;
        var arrowId = id('arrow', index);
        var deltaX = to.x - from.x;
        var deltaY = to.y - from.y;
        var scale = Math.min(Math.abs(deltaX) ? 90 / Math.abs(deltaX) : Infinity, Math.abs(deltaY) ? 30 / Math.abs(deltaY) : Infinity);
        var startX = from.x + deltaX * scale;
        var startY = from.y + deltaY * scale;
        var endX = to.x - deltaX * scale;
        var endY = to.y - deltaY * scale;
        boxElements[edge.from].boundElements.push({ id: arrowId, type: 'arrow' });
        boxElements[edge.to].boundElements.push({ id: arrowId, type: 'arrow' });
        elements.push({ type: 'arrow', version: 1, versionNonce: index + 3001, isDeleted: false, id: arrowId, fillStyle: 'solid', strokeWidth: 2, strokeStyle: edge.label === 'related' ? 'dashed' : 'solid', roughness: 0, opacity: 100, angle: 0, x: startX, y: startY, strokeColor: '#adb5bd', backgroundColor: 'transparent', width: endX - startX, height: endY - startY, seed: index + 4001, points: [[0, 0], [endX - startX, endY - startY]], startBinding: { elementId: fromId, focus: 0, gap: 5 }, endBinding: { elementId: toId, focus: 0, gap: 5 }, startArrowhead: null, endArrowhead: 'arrow', updated: Date.now(), link: null, locked: false });
      });
      var payload = { type: 'excalidraw', version: 2, source: 'problemrider', elements: elements, appState: { viewBackgroundColor: '#ffffff' }, files: {} };
      var blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
      var link = document.createElement('a');
      link.href = URL.createObjectURL(blob);
      var timestamp = new Date().toISOString().replace(/[-:]/g, '').replace(/\.\d{3}Z$/, 'Z');
      link.download = 'analysis-workbench-excalidraw-' + timestamp + '.excalidraw';
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(link.href);
    });

    var saveWorkbenchButton = document.querySelector('[data-analysis-trail-save]');
    if (saveWorkbenchButton) saveWorkbenchButton.addEventListener('click', function () {
      var payload = {
        format: 'problemrider-analysis-workbench',
        version: 1,
        exportedAt: new Date().toISOString(),
        trail: snapshot(trail)
      };
      var blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
      var link = document.createElement('a');
      link.href = URL.createObjectURL(blob);
      var timestamp = new Date().toISOString().replace(/[-:]/g, '').replace(/\.\d{3}Z$/, 'Z');
      link.download = 'analysis-workbench-' + timestamp + '.json';
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(link.href);
    });

    function validatedWorkbench(payload) {
      if (!payload || payload.format !== 'problemrider-analysis-workbench' || payload.version !== 1 || !payload.trail) throw new Error('Unsupported workbench file');
      var source = payload.trail;
      if (!Array.isArray(source.nodes) || !Array.isArray(source.edges)) throw new Error('Invalid workbench data');
      var allowedTypes = ['problem', 'symptom', 'root cause', 'solution'];
      var nodes = source.nodes.slice(0, maxNodes).filter(function (item) {
        return item && typeof item.id === 'string' && typeof item.title === 'string' && allowedTypes.indexOf(item.type) !== -1;
      }).map(function (item) {
        var custom = item.custom === true;
        var safeUrl = custom ? '#' : (typeof item.url === 'string' && /^\/(problems|solutions)\/[^/]+\.html(?:\?.*)?$/.test(item.url) ? item.url : '#');
        return { id: item.id.slice(0, 250), title: item.title.slice(0, 500), type: item.type, custom: custom || safeUrl === '#', url: safeUrl };
      });
      var ids = {};
      nodes.forEach(function (item) { ids[item.id] = true; });
      var edges = source.edges.slice(0, maxEdges).filter(function (item) {
        return item && ids[item.from] && ids[item.to] && typeof item.label === 'string';
      }).map(function (item) { return { from: item.from, to: item.to, label: item.label.slice(0, 80) }; });
      var positions = {};
      nodes.forEach(function (item) {
        var position = source.positions && source.positions[item.id];
        if (position && Number.isFinite(position.x) && Number.isFinite(position.y)) positions[item.id] = { x: position.x, y: position.y };
      });
      var pan = source.pan && Number.isFinite(source.pan.x) && Number.isFinite(source.pan.y) ? { x: source.pan.x, y: source.pan.y } : { x: 0, y: 300 };
      var result = { nodes: nodes, edges: edges, positions: positions, pan: pan };
      if (Number.isFinite(source.zoom)) result.zoom = Math.max(0.2, Math.min(3, source.zoom));
      return result;
    }

    var loadWorkbenchButton = document.querySelector('[data-analysis-trail-load]');
    var loadWorkbenchFile = document.querySelector('[data-analysis-trail-load-file]');
    if (loadWorkbenchButton) loadWorkbenchButton.addEventListener('click', function () { loadWorkbenchFile.click(); });
    if (loadWorkbenchFile) loadWorkbenchFile.addEventListener('change', function () {
      var file = loadWorkbenchFile.files && loadWorkbenchFile.files[0];
      if (!file) return;
      file.text().then(function (content) {
        var imported = validatedWorkbench(JSON.parse(content));
        rememberChange(trail);
        trail = imported;
        saveTrail(trail);
        render(trail);
      }).catch(function () {
        window.alert('This is not a valid ProblemRider Analysis Workbench file.');
      }).finally(function () { loadWorkbenchFile.value = ''; });
    });

    function restoreHistory(direction) {
      var history = getHistory();
      var source = direction === 'undo' ? history.undo : history.redo;
      var destination = direction === 'undo' ? history.redo : history.undo;
      if (!source.length) return;
      destination.push(snapshot(trail));
      trail = source.pop();
      saveHistory(history);
      saveTrail(trail);
      render(trail);
    }
    var undo = document.querySelector('[data-analysis-trail-undo]');
    var redo = document.querySelector('[data-analysis-trail-redo]');
    if (undo) undo.addEventListener('click', function () { restoreHistory('undo'); });
    if (redo) redo.addEventListener('click', function () { restoreHistory('redo'); });
  });
}());
