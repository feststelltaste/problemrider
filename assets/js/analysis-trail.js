(function () {
  'use strict';

  // Versioned storage prevents edges created by an older navigation-based
  // implementation from being mixed with the semantic causal graph.
  var storageKey = 'problemrider-analysis-trail-v4';
  var pendingKey = 'problemrider-analysis-trail-pending-edge-v4';
  var expandedKey = 'problemrider-analysis-trail-expanded-v1';
  var historyKey = 'problemrider-analysis-trail-history-v1';
  var maxNodes = 24;
  var maxEdges = 30;
  var namespace = 'http://www.w3.org/2000/svg';

  function getTrail() {
    try {
      var saved = JSON.parse(window.sessionStorage.getItem(storageKey));
      if (saved && Array.isArray(saved.nodes) && Array.isArray(saved.edges)) {
        saved.positions = saved.positions || {};
        return saved;
      }
      return { nodes: [], edges: [], positions: {} };
    } catch (error) {
      return { nodes: [], edges: [], positions: {} };
    }
  }

  function saveTrail(trail) {
    window.sessionStorage.setItem(storageKey, JSON.stringify(trail));
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
          if (heading.indexOf('Symptoms') === 0) return { label: 'causes', direction: 'forward', targetType: 'symptom' };
          if (heading.indexOf('Root Causes') === 0) return { label: 'causes', direction: 'reverse', targetType: 'root cause' };
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
    if (link.closest('.related-problems')) return { label: 'related', direction: 'forward', targetType: 'problem' };
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
      if (targetNode) targetNode.type = pending.targetType;
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
      (kind === 'addressed-problems' ? '.addressed-problems a[href]' : '');
    if (selector) {
      links = Array.prototype.slice.call(pageDocument.querySelectorAll(selector));
    } else {
      var headingStart = kind === 'symptoms' ? 'Symptoms' : 'Root Causes';
      var heading = Array.prototype.slice.call(pageDocument.querySelectorAll('h2')).filter(function (item) {
        return item.textContent.trim().indexOf(headingStart) === 0;
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

  function nodeX(type, width) {
    var positions = { 'root cause': 0.16, problem: 0.45, symptom: 0.72, solution: 0.88 };
    return Math.round(width * (positions[type] || 0.5));
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
    var currentNodeId = trail.nodes[trail.nodes.length - 1].id;
    var width = 1200;
    var height = 900;
    var zoom = Math.max(0.2, Math.min(3, trail.zoom || 0.75));
    var visibleWidth = width / zoom;
    var visibleHeight = height / zoom;
    var viewBoxX = (width - visibleWidth) / 2;
    var viewBoxY = (height - visibleHeight) / 2;
    var svg = svgElement('svg', { viewBox: viewBoxX + ' ' + viewBoxY + ' ' + visibleWidth + ' ' + visibleHeight, role: 'img', 'aria-label': 'Analysis navigation graph' });
    var defs = svgElement('defs');
    var marker = svgElement('marker', { id: 'analysis-trail-arrow', viewBox: '0 0 10 10', refX: '8', refY: '5', markerWidth: '8', markerHeight: '8', orient: 'auto' });
    marker.appendChild(svgElement('path', { d: 'M 0 0 L 10 5 L 0 10 z', fill: '#a0aec0' }));
    defs.appendChild(marker);
    svg.appendChild(defs);

    var groupedNodes = { symptom: [], problem: [], solution: [], 'root cause': [] };
    displayNodes.forEach(function (node) {
      (groupedNodes[node.type] || groupedNodes.problem).push(node);
    });
    var rowY = { symptom: 28, problem: 98, solution: 168, 'root cause': 238 };
    var positions = {};
    displayNodes.forEach(function (node, index) {
      var group = groupedNodes[node.type] || groupedNodes.problem;
      var groupIndex = group.indexOf(node);
      var defaultPosition = {
        x: Math.round(width * (groupIndex + 1) / (group.length + 1)),
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
      edgeInfo.path.setAttribute('d', 'M ' + startX + ' ' + startY + ' C ' + controlOneX + ' ' + controlOneY + ', ' + controlTwoX + ' ' + controlTwoY + ', ' + endX + ' ' + endY);
    }

    function updateNode(nodeId) {
      var elements = nodeElements[nodeId];
      var position = positions[nodeId];
      if (!elements || !position) return;
      elements.circle.setAttribute('cx', position.x);
      elements.circle.setAttribute('cy', position.y);
      elements.text.setAttribute('x', position.x);
      elements.text.setAttribute('y', position.y + 24);
      if (elements.removeCircle) {
        elements.removeCircle.setAttribute('cx', position.x + 10);
        elements.removeCircle.setAttribute('cy', position.y - 10);
        elements.removeIcon.setAttribute('x', position.x + 10);
        elements.removeIcon.setAttribute('y', position.y - 7.5);
      }
    }

    function updateGraph() {
      edgeElements.forEach(updateEdge);
      Object.keys(nodeElements).forEach(updateNode);
    }

    function hideNodeMenu() {
      if (nodeMenu) nodeMenu.remove();
      nodeMenu = null;
    }

    function scheduleMenuHide() {
      window.clearTimeout(menuHideTimer);
      menuHideTimer = window.setTimeout(hideNodeMenu, 250);
    }

    function navigateToReference(sourceNode, kind, reference) {
      var relationship = {
        symptoms: { label: 'causes', direction: 'forward', targetType: 'symptom' },
        causes: { label: 'causes', direction: 'reverse', targetType: 'root cause' },
        solutions: { label: 'addresses', direction: 'reverse', targetType: 'solution' },
        'addressed-problems': { label: 'addresses', direction: 'forward', targetType: 'problem' }
      }[kind];
      relationship.from = sourceNode.id;
      window.sessionStorage.setItem(pendingKey, JSON.stringify(relationship));
      window.location.href = reference.url;
    }

    function showNodeMenu(sourceNode) {
      window.clearTimeout(menuHideTimer);
      hideNodeMenu();
      var position = positions[sourceNode.id];
      if (!position) return;
      nodeMenu = document.createElement('div');
      nodeMenu.className = 'analysis-trail__node-menu';
      nodeMenu.setAttribute('aria-label', 'References for ' + sourceNode.title);
      var svgBounds = svg.getBoundingClientRect();
      var containerBounds = container.getBoundingClientRect();
      var left = svgBounds.left - containerBounds.left + (position.x - viewBoxX) * svgBounds.width / visibleWidth;
      var top = svgBounds.top - containerBounds.top + (position.y - viewBoxY) * svgBounds.height / visibleHeight;
      nodeMenu.style.left = left + 'px';
      nodeMenu.style.top = top + 'px';

      var menuActions = sourceNode.id.indexOf('solution:') === 0 ?
        [{ kind: 'addressed-problems', label: 'Addressed Problems' }] :
        [{ kind: 'symptoms', label: 'Symptoms' }, { kind: 'causes', label: 'Causes' }, { kind: 'solutions', label: 'Solutions' }];
      menuActions.forEach(function (menuAction) {
        var kind = menuAction.kind;
        var action = document.createElement('button');
        action.type = 'button';
        action.className = 'analysis-trail__node-menu-action analysis-trail__node-menu-action--' + kind;
        action.textContent = menuAction.label;
        action.addEventListener('click', function (event) {
          event.stopPropagation();
          var list = document.createElement('div');
          list.className = 'analysis-trail__node-menu-list';
          list.textContent = 'Loading…';
          var actionBounds = action.getBoundingClientRect();
          var menuBounds = nodeMenu.getBoundingClientRect();
          list.style.left = (actionBounds.left - menuBounds.left) + 'px';
          list.style.top = (actionBounds.bottom - menuBounds.top + 4) + 'px';
          nodeMenu.querySelectorAll('.analysis-trail__node-menu-list').forEach(function (item) { item.remove(); });
          nodeMenu.appendChild(list);
          referencesForNode(sourceNode, kind).then(function (references) {
            list.innerHTML = '';
            if (!references.length) {
              list.textContent = 'No references.';
              return;
            }
            references.forEach(function (reference) {
              var referenceButton = document.createElement('button');
              referenceButton.type = 'button';
              referenceButton.textContent = reference.title;
              referenceButton.addEventListener('click', function () {
                navigateToReference(sourceNode, kind, reference);
              });
              list.appendChild(referenceButton);
            });
          }).catch(function () { list.textContent = 'Could not load references.'; });
        });
        nodeMenu.appendChild(action);
      });
      nodeMenu.addEventListener('pointerenter', function () { window.clearTimeout(menuHideTimer); });
      nodeMenu.addEventListener('pointerleave', scheduleMenuHide);
      container.appendChild(nodeMenu);
    }

    trail.edges.forEach(function (edge) {
      var from = positions[edge.from];
      var to = positions[edge.to];
      if (!from || !to) return;
      var edgeClass = 'analysis-trail__edge analysis-trail__edge--' + edge.label.replace(/\s+/g, '-');
      var path = svgElement('path', { class: edgeClass, 'marker-end': 'url(#analysis-trail-arrow)' });
      svg.appendChild(path);
      var edgeInfo = { edge: edge, path: path };
      edgeElements.push(edgeInfo);
      updateEdge(edgeInfo);
    });
    displayNodes.forEach(function (node) {
      var position = positions[node.id];
      var link = svgElement('a', { href: node.url, class: 'analysis-trail__node-link' });
      var circle = svgElement('circle', { cx: position.x, cy: position.y, r: '10', class: 'analysis-trail__node analysis-trail__node--' + node.type.replace(/\s+/g, '-') + (node.id === currentNodeId ? ' is-current' : '') });
      var title = svgElement('title');
      title.textContent = node.title;
      circle.appendChild(title);
      link.appendChild(circle);
      var text = svgElement('text', { x: position.x, y: position.y + 24, class: 'analysis-trail__node-label', 'text-anchor': 'middle' });
      text.textContent = node.title.length > 16 ? node.title.slice(0, 15) + '…' : node.title;
      link.appendChild(text);
      svg.appendChild(link);
      nodeElements[node.id] = { circle: circle, text: text };

      // Every node can be removed; any connected arrows are removed as well.
      {
        var remove = svgElement('g', {
          class: 'analysis-trail__remove-node',
          role: 'button',
          tabindex: '0',
          'aria-label': 'Remove ' + node.title + ' from the analysis trail'
        });
        remove.appendChild(svgElement('circle', { cx: position.x + 10, cy: position.y - 10, r: '6' }));
        var removeIcon = svgElement('text', { x: position.x + 10, y: position.y - 7.5, 'text-anchor': 'middle' });
        removeIcon.textContent = '×';
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
        svg.appendChild(remove);
        nodeElements[node.id].removeCircle = remove.firstChild;
        nodeElements[node.id].removeIcon = removeIcon;
      }
    });
    var draggedNodeId = null;
    var dragged = false;
    var suppressClick = false;

    function svgPoint(event) {
      var bounds = svg.getBoundingClientRect();
      return {
        x: viewBoxX + (event.clientX - bounds.left) * visibleWidth / bounds.width,
        y: viewBoxY + (event.clientY - bounds.top) * visibleHeight / bounds.height
      };
    }

    Object.keys(nodeElements).forEach(function (nodeId) {
      var nodeElement = nodeElements[nodeId];
      nodeElement.circle.addEventListener('pointerdown', function (event) {
        if (event.button !== 0) return;
        draggedNodeId = nodeId;
        dragged = false;
        svg.setPointerCapture(event.pointerId);
      });
      nodeElement.circle.addEventListener('pointerup', function (event) {
        if (draggedNodeId === nodeId && !dragged) {
          event.preventDefault();
          window.location.href = nodeElement.circle.parentElement.getAttribute('href');
        }
      });
      nodeElement.circle.addEventListener('pointerenter', function () {
        showNodeMenu(trail.nodes.filter(function (item) { return item.id === nodeId; })[0]);
      });
      nodeElement.circle.addEventListener('pointerleave', scheduleMenuHide);
      nodeElement.circle.parentElement.addEventListener('click', function (event) {
        if (suppressClick) {
          event.preventDefault();
          suppressClick = false;
        }
      });
    });

    svg.addEventListener('pointermove', function (event) {
      if (!draggedNodeId) return;
      var point = svgPoint(event);
      positions[draggedNodeId] = {
        x: Math.max(14, Math.min(width - 14, point.x)),
        y: Math.max(18, Math.min(height - 18, point.y))
      };
      dragged = true;
      updateGraph();
    });
    svg.addEventListener('pointerup', function (event) {
      if (!draggedNodeId) return;
      if (dragged) {
        rememberChange(trail);
        trail.positions[draggedNodeId] = positions[draggedNodeId];
        saveTrail(trail);
        suppressClick = true;
      }
      if (svg.hasPointerCapture(event.pointerId)) svg.releasePointerCapture(event.pointerId);
      draggedNodeId = null;
    });
    container.appendChild(svg);
  }

  document.addEventListener('DOMContentLoaded', function () {
    var trail = getTrail();
    var node = currentNode();
    addCurrentNode(trail, node);
    addPendingEdge(trail, node);
    saveTrail(trail);
    render(trail);

    var trailLayout = document.querySelector('.page-with-analysis-trail');
    var modeButton = document.querySelector('[data-analysis-trail-mode]');
    var openButton = document.querySelector('[data-analysis-trail-open]');
    function setExpanded(isExpanded) {
      if (!trailLayout) return;
      trailLayout.classList.toggle('is-analysis-trail-expanded', isExpanded);
      document.body.classList.toggle('analysis-trail-expanded', isExpanded);
      if (modeButton) modeButton.setAttribute('aria-pressed', String(isExpanded));
      window.sessionStorage.setItem(expandedKey, String(isExpanded));
    }
    if (window.sessionStorage.getItem(expandedKey) === 'true') setExpanded(true);
    if (modeButton) modeButton.addEventListener('click', function () {
      setExpanded(!trailLayout.classList.contains('is-analysis-trail-expanded'));
    });
    if (openButton) openButton.addEventListener('click', function () { setExpanded(true); });

    document.addEventListener('click', function (event) {
      var link = event.target.closest('a[href]');
      if (!link || link.target === '_blank' || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey || !node) return;
      var target;
      try { target = new URL(link.href, window.location.href); } catch (error) { return; }
      if (target.origin !== window.location.origin || !/\/(problems|solutions)\/[^/]+\.html$/.test(target.pathname)) return;
      var edge = edgeForLink(link);
      if (!edge) return;
      edge.from = node.id;
      window.sessionStorage.setItem(pendingKey, JSON.stringify(edge));
    });

    var reset = document.querySelector('[data-analysis-trail-reset]');
    if (reset) reset.addEventListener('click', function () {
      rememberChange(trail);
      window.sessionStorage.removeItem(storageKey);
      window.sessionStorage.removeItem(pendingKey);
      var freshTrail = { nodes: [], edges: [], positions: {}, zoom: 0.75 };
      if (node) addCurrentNode(freshTrail, node);
      trail = freshTrail;
      saveTrail(freshTrail);
      render(freshTrail);
    });

    function changeZoom(change) {
      rememberChange(trail);
      trail.zoom = Math.max(0.2, Math.min(3, (trail.zoom || 0.75) + change));
      saveTrail(trail);
      render(trail);
    }
    var zoomIn = document.querySelector('[data-analysis-trail-zoom-in]');
    var zoomOut = document.querySelector('[data-analysis-trail-zoom-out]');
    if (zoomIn) zoomIn.addEventListener('click', function () { changeZoom(0.1); });
    if (zoomOut) zoomOut.addEventListener('click', function () { changeZoom(-0.1); });

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
