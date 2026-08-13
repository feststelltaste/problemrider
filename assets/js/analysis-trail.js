(function () {
  'use strict';

  // Versioned storage prevents edges created by an older navigation-based
  // implementation from being mixed with the semantic causal graph.
  var storageKey = 'problemrider-analysis-trail-v12';
  var pendingKey = 'problemrider-analysis-trail-pending-edge-v12';
  var expandedKey = 'problemrider-analysis-trail-expanded-v1';
  var historyKey = 'problemrider-analysis-trail-history-v1';
  var speedNavKey = 'problemrider-analysis-trail-speed-nav-v1';
  var maxNodes = 200;
  var maxEdges = 30;
  var namespace = 'http://www.w3.org/2000/svg';
  var spacePressed = false;

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

  function nodeFromReference(reference) {
    var match = reference.url.match(/\/(problems|solutions)\/([^/]+)\.html/);
    if (!match) return null;
    return {
      id: (match[1] === 'solutions' ? 'solution:' : 'problem:') + match[2],
      title: reference.title,
      type: match[1] === 'solutions' ? 'solution' : 'problem',
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
    var currentNodeId = trail.nodes[trail.nodes.length - 1].id;
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
      elements.labelLines.forEach(function (line, index) {
        line.setAttribute('x', position.x);
        line.setAttribute('y', position.y + 24 + index * 10);
      });
      if (elements.removeIcon) {
        elements.removeIcon.setAttribute('x', position.x);
        elements.removeIcon.setAttribute('y', position.y - 17);
      }
    }

    function updateGraph() {
      edgeElements.forEach(updateEdge);
      Object.keys(nodeElements).forEach(updateNode);
    }

    function updateViewBox() {
      svg.setAttribute('viewBox', viewBoxX + ' ' + viewBoxY + ' ' + visibleWidth + ' ' + visibleHeight);
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
        symptoms: { label: 'causes', direction: 'reverse', targetType: 'symptom' },
        causes: { label: 'causes', direction: 'forward', targetType: 'root cause' },
        solutions: { label: 'addresses', direction: 'reverse', targetType: 'solution' },
        'addressed-problems': { label: 'addresses', direction: 'forward', targetType: 'problem' },
        'similar-solutions': { label: 'related', direction: 'forward', targetType: 'solution' },
        'similar-problems': { label: 'related', direction: 'forward', targetType: 'problem' }
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
        [{ kind: 'addressed-problems', label: 'Addressed Problems' }, { kind: 'similar-solutions', label: 'Similar Solutions' }] :
        [{ kind: 'symptoms', label: 'Symptoms' }, { kind: 'causes', label: 'Causes' }, { kind: 'solutions', label: 'Solutions' }, { kind: 'similar-problems', label: 'Similar Problems' }];
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
                  var referenceNode = nodeFromReference(reference);
                  if (!referenceNode) return;
                  addCurrentNode(trail, referenceNode);
                  var relation = {
                    symptoms: { from: referenceNode.id, to: sourceNode.id, label: 'causes' },
                    causes: { from: sourceNode.id, to: referenceNode.id, label: 'causes' },
                    solutions: { from: referenceNode.id, to: sourceNode.id, label: 'addresses' },
                    'addressed-problems': { from: sourceNode.id, to: referenceNode.id, label: 'addresses' }
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
      var pathAttributes = { class: edgeClass };
      if (edge.label !== 'related') pathAttributes['marker-end'] = 'url(#analysis-trail-arrow)';
      var path = svgElement('path', pathAttributes);
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
      var lines = labelLines(node.title, 20);
      lines.forEach(function (line, index) {
        var span = svgElement('tspan', { x: position.x, y: position.y + 24 + index * 10 });
        span.textContent = line;
        text.appendChild(span);
      });
      link.appendChild(text);
      svg.appendChild(link);
      nodeElements[node.id] = { circle: circle, text: text, labelLines: Array.prototype.slice.call(text.querySelectorAll('tspan')) };

      // Every node can be removed; any connected arrows are removed as well.
      {
        var remove = svgElement('g', {
          class: 'analysis-trail__remove-node',
          role: 'button',
          tabindex: '0',
          'aria-label': 'Remove ' + node.title + ' from the analysis trail'
        });
        var removeIcon = svgElement('text', { x: position.x, y: position.y - 17, 'text-anchor': 'middle' });
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
        link.appendChild(remove);
        nodeElements[node.id].removeIcon = removeIcon;
      }
    });
    var draggedNodeId = null;
    var dragged = false;
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

    Object.keys(nodeElements).forEach(function (nodeId) {
      var nodeElement = nodeElements[nodeId];
      nodeElement.circle.addEventListener('pointerdown', function (event) {
        if (event.button !== 0) return;
        if (spacePressed) return;
        event.stopPropagation();
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

    svg.addEventListener('pointerdown', function (event) {
      var isMiddleMouse = event.button === 1;
      var isSpacePan = event.button === 0 && spacePressed;
      if (!isMiddleMouse && !isSpacePan && event.target !== svg) return;
      if (!isMiddleMouse && !isSpacePan && event.button !== 0) return;
      event.preventDefault();
      hideNodeMenu();
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
        positions[draggedNodeId] = {
          x: Math.max(14, Math.min(width - 14, point.x)),
          y: Math.max(18, Math.min(height - 18, point.y))
        };
        dragged = true;
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
      if (draggedNodeId && dragged) {
        rememberChange(trail);
        trail.positions[draggedNodeId] = positions[draggedNodeId];
        saveTrail(trail);
        suppressClick = true;
      }
      if (svg.hasPointerCapture(event.pointerId)) svg.releasePointerCapture(event.pointerId);
      draggedNodeId = null;
      if (panning && panChanged) {
        saveTrail(trail);
        suppressClick = true;
      }
      panning = false;
      svg.classList.remove('is-panning');
    });
    container.appendChild(svg);
  }

  document.addEventListener('DOMContentLoaded', function () {
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
