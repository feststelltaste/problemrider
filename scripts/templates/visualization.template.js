const graph = {GRAPH_DATA};

const width = Math.min(1200, window.innerWidth * 0.9);
const height = window.innerHeight - 60; // Account for header

const categoryColors = {
    // 15 core categories with vibrant, distinct colors
    'Architecture': '#3498db',    // Blue
    'Business': '#e74c3c',        // Red
    'Code': '#f39c12',            // Orange
    'Communication': '#9b59b6',   // Purple
    'Culture': '#e67e22',         // Dark Orange
    'Database': '#2ecc71',        // Green
    'Dependencies': '#16a085',    // Teal
    'Management': '#e91e63',      // Pink
    'Operations': '#34495e',      // Dark Blue-Gray
    'Performance': '#f1c40f',     // Yellow
    'Process': '#27ae60',         // Dark Green
    'Requirements': '#8e44ad',    // Dark Purple
    'Security': '#c0392b',        // Dark Red
    'Team': '#1abc9c',            // Turquoise
    'Testing': '#ff6b35'          // Red-Orange
};

const color = (category) => categoryColors[category] || '#6c757d';

const simulation = d3.forceSimulation(graph.nodes)
    .force("link", d3.forceLink(graph.links).id(d => d.id).distance(100))
    .force("charge", d3.forceManyBody().strength(-1000))
    .force("center", d3.forceCenter(width / 2, height / 2))
    .force("x", d3.forceX(width / 2).strength(0.1))
    .force("y", d3.forceY(height / 2).strength(0.1))
    .force("collision", d3.forceCollide().radius(30));

const svg = d3.select("#visualization-container").append("svg")
    .attr("width", width)
    .attr("height", height)
    .call(d3.zoom()
        .extent([[0, 0], [width, height]])
        .scaleExtent([0.1, 8])
        .on("zoom", function (event) {
            container.attr("transform", event.transform);
        }));

// Container for all elements that should be zoomed/panned
const container = svg.append("g");

// Arrowhead marker with dynamic position
svg.append('defs').append('marker')
    .attr('id', 'arrowhead')
    .attr('viewBox', '-0 -5 10 10')
    .attr('refX', 35)
    .attr('refY', 0)
    .attr('orient', 'auto')
    .attr('markerWidth', 8)
    .attr('markerHeight', 8)
    .attr('xoverflow', 'visible')
    .append('svg:path')
    .attr('d', 'M 0,-5 L 10 ,0 L 0,5')
    .attr('fill', '#999')
    .style('stroke', 'none');

const link = container.append('g')
    .attr('class', 'link')
    .selectAll('line')
    .data(graph.links)
    .enter().append('line')
    .attr('stroke-width', function (d) { return Math.sqrt(d.value || 1); })
    .attr('marker-end', 'url(#arrowhead)');

const node = container.append('g')
    .attr('class', 'nodes')
    .selectAll('g')
    .data(graph.nodes)
    .enter().append('g')
    .attr('class', 'node')
    .call(d3.drag()
        .on('start', dragstarted)
        .on('drag', dragged)
        .on('end', dragended));

// Variables to handle selection and opacity
let selectedNode = null;
const nonActiveOpacity = 0.2;

node.append('circle')
    .attr('r', function (d) { return d.size || 10; })
    .attr('fill', function (d) { return color(d.category); });

node.append('text')
    .text(function (d) { return d.title; })
    .attr('dy', -15);

const tooltip = d3.select(".tooltip");

// Update event handlers and add 'click' handler.
node.on("mouseover", (event, d) => {
    tooltip.transition().style("opacity", 1);
    tooltip.html(`<strong>${d.title}</strong><br/><em>${d.category}</em><br/>${d.description}`)
        .style("left", (event.pageX + 15) + "px")
        .style("top", (event.pageY - 28) + "px");
})
    .on("mouseout", () => {
        tooltip.transition().style("opacity", 0);
    })
    .on("click", (event, d) => {
        event.stopPropagation(); // Prevent click from propagating to background (SVG)
        // If clicking the already selected node, deselect. Otherwise, select.
        selectedNode = (selectedNode && selectedNode.id === d.id) ? null : d;
        if (pathManager.isCreatingPath) {
            selectedNode = d;
            if (pathManager.isAddingNode) {
                pathManager.currentPath.add(selectedNode.id);
            }
            else if (pathManager.isRemovingNode) {
                pathManager.currentPath.delete(selectedNode.id);
            }
        }
        updateStyles();
    })
    .on("dblclick", (event, d) => {
        event.stopPropagation();
        openModal(d);
    });

// Click on background to deselect all
svg.on('click', function () {
    if (!pathManager.isCreatingPath) { //To "disable" background click when creating a path
        selectedNode = null;
        updateStyles();
    }
});

const categories = [...new Set(graph.nodes.map(d => d.category))].sort();
const legend = d3.select(".legend");

categories.forEach(category => {
    const legendItem = legend.append("div").attr("class", "legend-item");
    legendItem.append("div")
        .attr("class", "legend-color")
        .style("background-color", color(category));
    legendItem.append("span").text(category);
});

simulation
    .on("tick", () => {
        link
            .attr("x1", d => d.source.x)
            .attr("y1", d => d.source.y)
            .attr("x2", d => d.target.x)
            .attr("y2", d => d.target.y);

        node
            .attr("transform", d => `translate(${d.x}, ${d.y})`);
    });

// Central function to update node and link opacity
function updateStyles() {
    if (!pathManager.isCreatingPath) {
        node.style('opacity', 1);
        link.style('stroke-opacity', 0.6)
            .style('stroke', '#adb5bd')
            .style('filter', 'none');

        if (selectedNode) {
            // If a node is selected
            node.style('opacity', d => (d.id === selectedNode.id ? 1 : nonActiveOpacity));
            link.style('stroke-opacity', l => (l.source.id === selectedNode.id || l.target.id === selectedNode.id) ? 1 : nonActiveOpacity)
                .style('stroke', l => (l.source.id === selectedNode.id || l.target.id === selectedNode.id) ? '#6c757d' : '#adb5bd');
        }
        return;
    }

    if (!selectedNode) return;

    node.select('circle')
        // Highlight selected node in green
        // Highlight nodes in path in black
        // Rest in white
        .style('stroke', d => {
            if (d.id === selectedNode.id) return 'green';
            else if (pathManager.currentPath.has(d.id)) return 'black';
            else return '#fff';
        })
        //Set stroke-width to 3 to all nodes in path, otherwise 1.5.
        .style('stroke-width', d => { pathManager.currentPath.has(d.id) || d.id === selectedNode.id ? 3 : 1.5 });

    // Enable to select/add nodes in path or neighbors of selected node.
    //const neighboringNodes = getNeighboringNodes(selectedNode.id);
    //node.style('pointer-events', d => { (pathManager.currentPath.has(d.id) || neighboringNodes.has(d.id) || d.id === selectedNode.id) ? 'all' : 'none'; });

    // Set opacity 1 to all nodes in path.
    node.style('opacity', d => pathManager.currentPath.has(d.id) ? 1 : nonActiveOpacity);

    // Set stroke color and opacity black to the links connected to the selected node.
    // Set Stroke color lime to all links among the nodes in path.
    link.style('filter', l => (pathManager.currentPath.has(l.source.id) && pathManager.currentPath.has(l.target.id)) ? 'drop-shadow(0px 0px 3px lime)' : 'none')
        .style('stroke-opacity', l => {
            const inPath = pathManager.currentPath.has(l.source.id) && pathManager.currentPath.has(l.target.id);
            const connectedToSelected = l.source.id === selectedNode.id || l.target.id === selectedNode.id;
            return inPath || connectedToSelected ? 1 : nonActiveOpacity; // Full opacity
        })
        .style('stroke', l => {
            const inPath = pathManager.currentPath.has(l.source.id) && pathManager.currentPath.has(l.target.id);
            const connectedToSelected = l.source.id === selectedNode.id || l.target.id === selectedNode.id;
            return inPath ? 'lime' : (connectedToSelected ? 'black' : '#adb5bd'); // Lime for path, black for connected, gray otherwise
        });
}

function dragstarted(event, d) {
    if (!event.active) simulation.alphaTarget(0.3).restart();
    d.fx = d.x;
    d.fy = d.y;
}

function dragged(event, d) {
    d.fx = event.x;
    d.fy = event.y;
}

function dragended(event, d) {
    if (!event.active) simulation.alphaTarget(0);
    d.fx = null;
    d.fy = null;
}

// Create Path Object Literal
const pathManager = {
    _isCreatingPath: false,
    _isAddingNode: false,
    _isRemovingNode: false,
    _storageKey: 'allMyPaths',
    currentPath: new Set(),
    selectedNode: null,

    get isCreatingPath() {
        return this._isCreatingPath;
    },
    set isCreatingPath(val) {
        this._isCreatingPath = val;
        if (!val) {
            this._isAddingNode = false;
            this._isRemovingNode = false;
        }
    },
    get isAddingNode() {
        return this._isAddingNode;
    },
    set isAddingNode(val) {
        if (this._isCreatingPath) {
            this._isAddingNode = val;
            if (val) this._isRemovingNode = false;
        } else {
            this._isAddingNode = false;
        }
    },
    get isRemovingNode() {
        return this._isRemovingNode;
    },
    set isRemovingNode(val) {
        if (this._isCreatingPath) {
            this._isRemovingNode = val;
            if (val) this._isAddingNode = false;
        } else {
            this._isRemovingNode = false;
        }
    },
    get localStoragePaths() {
        const pathsString = localStorage.getItem(this._storageKey);
        return pathsString ? JSON.parse(pathsString) : [];
    },
    set localStoragePaths(pathsArray) {
        localStorage.setItem(this._storageKey, JSON.stringify(pathsArray));
    },
    /**
     * Add a new path to the existing array in localStorage.
     * @param {object} newPath - The new path to add.
     */
    addPathToLocalStorage: function (newPath) {
        const currentPaths = this.localStoragePaths;
        currentPaths.push(newPath);
        this.localStoragePaths = currentPaths;
    },
    /**
     * Remove a path by its name.
     * @param {string} pathNameToRemove - The name of the path to remove.
     */
    removePathFromLocalStorage: function (pathNameToRemove) {
        let currentPaths = this.localStoragePaths;
        currentPaths = currentPaths.filter(path => path.name !== pathNameToRemove);
        this.localStoragePaths = currentPaths;
    },
    outCreateMode: function () {
        this.isCreatingPath = false;
        selectedNode = null;
        this.currentPath.clear();
        updateStyles();
    },
    getPathfromLocalStorage: function (pathName) {
        const currentPaths = this.localStoragePaths;
        return currentPaths.filter(path => path.name === pathName)[0];
    },
    updatePathsInDom: function () {
        const paths = this.localStoragePaths;
        pathListContainer.innerHTML = ''; // Clear existing paths
        if (paths.length === 0) {
            pathListContainer.innerHTML = '<em>No saved paths.</em>';
            return;
        }
        paths.forEach(path => {
            const pathItem = document.createElement('div');
            pathItem.className = 'control-item';
            pathItem.innerHTML = `
                <span class="control-icon">📍</span>
                <span class="control-text" data-path-name="${path.name}">${path.name}</span>
                <span class="delete-path-btn" title="Delete Path" data-path-name="${path.name}">🚫</span>
                <span class="export-path-btn" title="Export" data-path-name="${path.name}">📥</span>
            `;
            pathListContainer.appendChild(pathItem);
        });
    },
    exportPath: function (pathName) {
        const path = this.getPathfromLocalStorage(pathName);
        const blob = new Blob([JSON.stringify(path, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `${pathName}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    },
    async importPath() {
        try {
            const file = await new Promise(resolve => {
                const input = document.createElement('input');
                input.type = 'file';
                input.accept = '.json';
                input.onchange = () => resolve(input.files[0]);
                input.click();
            });

            if (!file) return;

            const fileContent = await new Promise((resolve, reject) => {
                const reader = new FileReader();
                reader.onload = event => resolve(event.target.result);
                reader.onerror = error => reject(error);
                reader.readAsText(file);
            });

            this.addPathToLocalStorage(JSON.parse(fileContent));
            this.updatePathsInDom();

        } catch (error) {
            console.error("Import failed:", error);
            alert("The file could not be imported. Make sure it is a valid JSON.");
        }
    }
};

// HTML UI Event Listeners
const createPathButton = document.querySelector('#create-path-btn');
const importPathButton = document.querySelector('#import-path-btn');
const pathCreateOnTooltip = document.querySelector('.path-creator-on-tooltip');
const addNodeButton = document.querySelector('#add-node-btn');
const removeNodeButton = document.querySelector('#remove-node-btn');
const savePathButton = document.querySelector('#save-path-btn');
const cancelPathButton = document.querySelector('#cancel-path-btn');
const pathListContainer = document.querySelector('#path-list-container');

createPathButton.addEventListener('click', () => {
    createPathButton.disabled = true;
    addNodeButton.disabled = true;
    pathManager.isCreatingPath = true;
    pathManager.isAddingNode = true;
    pathManager.currentPath.clear();
    selectedNode = null;
    updateStyles();
    pathCreateOnTooltip.style.visibility = 'visible';
});
importPathButton.addEventListener('click', () => {
    pathManager.importPath();
});
addNodeButton.addEventListener('click', () => {
    pathManager.isAddingNode = true;
    addNodeButton.disabled = true;
    removeNodeButton.disabled = false;
    pathCreateOnTooltip.style.backgroundColor = 'lime'; // Light green background
});
removeNodeButton.addEventListener('click', () => {
    pathManager.isRemovingNode = true;
    removeNodeButton.disabled = true;
    addNodeButton.disabled = false;
    pathCreateOnTooltip.style.backgroundColor = '#fa858fff'; // Light red background
});
savePathButton.addEventListener('click', () => {
    if (pathManager.currentPath.size < 2) {
        alert('Please select at least two nodes to create a path.');
        return;
    }
    const pathArray = Array.from(pathManager.currentPath);
    const pathName = prompt('Enter a name for the new path:');
    if (pathName) {
        // Save the path (e.g., send to server or store locally)
        pathManager.addPathToLocalStorage({ name: pathName, nodes: pathArray });
        // Update the UI to show the new path
        pathManager.updatePathsInDom();
    }
});
cancelPathButton.addEventListener('click', () => {
    pathManager.outCreateMode();
    pathCreateOnTooltip.style.visibility = 'hidden';
    createPathButton.disabled = false;
});
pathListContainer.addEventListener('click', (event) => {
    // Check if a DELETE BUTTON was clicked
    const deleteButton = event.target.closest('.delete-path-btn');
    if (deleteButton) {
        const pathNameToDelete = deleteButton.dataset.pathName;
        pathManager.removePathFromLocalStorage(pathNameToDelete);
        pathManager.updatePathsInDom();
        return;
    }

    // Check if a PATH NAME was clicked
    const pathNameLink = event.target.closest('.control-text');
    if (pathNameLink) {
        event.preventDefault();
        const pathNameToLoad = pathNameLink.dataset.pathName;
        console.log(`Loading path: ${pathNameToLoad}`);

        const nodesInPath = pathManager.getPathfromLocalStorage(pathNameToLoad).nodes;

        pathManager.currentPath = new Set(nodesInPath);
        createPathButton.disabled = true;
        addNodeButton.disabled = true;
        pathManager.isCreatingPath = true;
        pathManager.isAddingNode = true;
        pathCreateOnTooltip.style.visibility = 'visible';
        selectedNode = nodesInPath[0];
        updateStyles();
        return
    }
    // Check if an EXPORT BUTTON was clicked
    const exportButton = event.target.closest('.export-path-btn');
    if (exportButton) {
        const pathNameToExport = exportButton.dataset.pathName;
        pathManager.exportPath(pathNameToExport);
        return;
    }
});

document.addEventListener('DOMContentLoaded', () => {
    //Load existing paths from localStorage and display them
    pathManager.updatePathsInDom();
});

// Modal functionality
const modal = document.getElementById('modal');
const modalTitle = document.getElementById('modal-title');
const modalCategory = document.getElementById('modal-category');
const modalBody = document.getElementById('modal-body');
const modalClose = document.getElementById('modal-close');

function openModal(nodeData) {
    modalTitle.textContent = nodeData.title;
    modalCategory.textContent = nodeData.category;
    modalBody.innerHTML = '<div class="loading">Loading problem details...</div>';
    modal.style.display = 'block';

    // Update URL hash when opening modal
    window.history.replaceState(null, null, '#' + nodeData.id);

    // Fetch the generated HTML file content
    const problemPath = nodeData.id.replace('.md', '.html');
    fetch(`problems/${problemPath}`)
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.text();
        })
        .then(content => {
            displayHtmlContent(content, nodeData);
        })
        .catch(error => {
            console.error('Error fetching problem content:', error);
            modalBody.innerHTML = `
                <div class="error">
                    <h3>Error Loading Content</h3>
                    <p>Could not load the full problem description.</p>
                    <div class="modal-description">${nodeData.description}</div>
                </div>
            `;
        });
}

function displayHtmlContent(content, nodeData) {
    // Parse the Jekyll-generated HTML to extract the main content
    const parser = new DOMParser();
    const doc = parser.parseFromString(content, 'text/html');

    // Find the main content area - Jekyll typically puts content in .page-content or main
    let mainContent = doc.querySelector('.page-content main') ||
        doc.querySelector('main') ||
        doc.querySelector('.page-content') ||
        doc.querySelector('article');

    if (!mainContent) {
        // Fallback: try to find content by looking for common patterns
        mainContent = doc.querySelector('div[class*="content"]') || doc.body;
    }

    let html = '';
    if (mainContent) {
        html = mainContent.innerHTML;

        // Convert relative markdown links to problem-link spans for modal navigation
        html = html.replace(/href="([^"]*\.md)"/g, function (match, mdFile) {
            const filename = mdFile.split('/').pop();
            return `class="problem-link" data-problem="${filename}"`;
        });

        // Also handle relative problem links
        html = html.replace(/href="\/problems\/([^"]*)\.html"/g, function (match, problemName) {
            return `class="problem-link" data-problem="${problemName}.md"`;
        });

        // Remove any Jekyll-specific elements we don't want
        html = html.replace(/<div class="page-header">.*?<\/div>/gs, '');
    } else {
        html = '<p>Could not extract content from the page.</p>';
    }

    modalBody.innerHTML = `
        ${html}
    `;

    // Add click handlers for problem links
    const problemLinks = modalBody.querySelectorAll('.problem-link');
    problemLinks.forEach(link => {
        link.addEventListener('click', function (e) {
            e.preventDefault();
            const problemFile = this.getAttribute('data-problem');
            const targetNode = graph.nodes.find(n => n.id === problemFile);
            if (targetNode) {
                openModal(targetNode);
            }
        });
    });

    // Convert regular links to open in new tab
    const externalLinks = modalBody.querySelectorAll('a:not(.problem-link)');
    externalLinks.forEach(link => {
        link.setAttribute('target', '_blank');
    });
}

function closeModal() {
    modal.style.display = 'none';
    // Clear URL hash when closing modal
    window.history.replaceState(null, null, window.location.pathname);
}

// Event listeners
modalClose.onclick = closeModal;
window.onclick = function (event) {
    if (event.target === modal) {
        closeModal();
    }
};

// Escape key to close modal
document.addEventListener('keydown', function (event) {
    if (event.key === 'Escape' && modal.style.display === 'block') {
        closeModal();
    }
});

// Function to highlight and center a specific node
function focusNode(nodeId) {
    const targetNode = graph.nodes.find(n => n.id === nodeId);
    if (!targetNode) return false;

    // Center the node
    const scale = d3.zoomTransform(svg.node()).k || 1;
    const x = -targetNode.x * scale + width / 2;
    const y = -targetNode.y * scale + height / 2;

    svg.transition()
        .duration(750)
        .call(d3.zoom().transform, d3.zoomIdentity.translate(x, y).scale(scale));

    // Highlight the node temporarily
    const nodeElement = node.filter(d => d.id === nodeId).select('circle');
    const originalColor = nodeElement.attr('fill');

    nodeElement
        .transition()
        .duration(200)
        .attr('fill', '#ff6b6b')
        .attr('r', function (d) { return (d.size || 10) * 1.5; })
        .transition()
        .duration(1000)
        .attr('fill', originalColor)
        .attr('r', function (d) { return d.size || 10; });

    return true;
}

// Handle URL fragments on page load
function handleUrlFragment() {
    const hash = window.location.hash.substring(1); // Remove the #
    if (hash) {
        // Wait for the simulation to settle before focusing
        setTimeout(() => {
            if (!focusNode(hash)) {
                console.warn('Node not found for hash:', hash);
            }
        }, 1000);
    }
}

// Initialize URL fragment handling after simulation starts
simulation.on('end', handleUrlFragment);

// Also handle hash changes during navigation
window.addEventListener('hashchange', handleUrlFragment);