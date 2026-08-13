# Problem Landscape Visualization Generator

This document explains how the `create_visualization.py` script works and what files it generates.

## Overview

The script generates an interactive D3.js-based network visualization of legacy system problems and their relationships. It processes markdown files from the `_problems` directory and creates a web-based interface for exploring problem patterns.

## How It Works

1. **Scans Problem Files**: Processes all `.md` files in the `_problems` directory
2. **Parses Metadata**: Extracts YAML frontmatter to get problem metadata (title, description, category)
3. **Analyzes Relationships**: Examines markdown content to find links between problems in "Symptoms" and "Root Causes" sections
4. **Creates Graph Structure**: Builds a directed graph where:
   - **Nodes** represent individual problems
   - **Edges** represent causal relationships (symptom → cause)
   - **Node sizes** reflect how many other problems reference them (incoming connections)
5. **Generates Output Files**: Creates two files for the Jekyll-based website

## Generated Files

### `landscape.html` (Project Root)
- Jekyll page with visualization interface
- Contains embedded HTML structure, CSS styling, and DOM elements
- Loads from `scripts/templates/landscape.template.html`
- Includes interactive controls, modal dialogs, and legend

### `assets/js/visualization.js` (Project Root)
- D3.js JavaScript file containing:
  - Complete graph data (nodes and links) as JSON
  - Force simulation configuration
  - Interactive event handlers
  - Modal functionality for problem details
  - URL fragment support for direct linking

## Template System

The script uses a template-based approach for better maintainability with two separate template files:

### HTML Template
- **Location**: `scripts/templates/landscape.template.html`
- **Purpose**: Contains Jekyll page structure, CSS styling, and DOM elements

### JavaScript Template  
- **Location**: `scripts/templates/visualization.template.js`
- **Purpose**: Contains all D3.js visualization logic and interactions
- **Placeholder**: `{GRAPH_DATA}` gets replaced with actual problem data

## Visualization Features

### Interactive Network Graph
- **Zoom/Pan**: Mouse wheel to zoom, drag to pan the entire graph
- **Node Dragging**: Drag individual nodes to reposition them
- **Force Simulation**: Automatic layout using D3.js physics simulation

### Visual Elements
- **Color Coding**: Problems are colored by category with a legend
- **Node Sizing**: Larger nodes indicate problems referenced by many others
- **Directed Edges**: Arrows show causal relationships between problems
- **Hover Effects**: Tooltips show problem details and highlight connections

### Problem Details
- **Double-click Modal**: Opens detailed problem description in overlay
- **Cross-navigation**: Links between problems work within the modal
- **URL Fragments**: Direct links to specific problems (e.g., `#problem-name.md`)
- **External Links**: Regular links open in new tabs

### Categories and Legend
The visualization supports multiple problem categories with distinct colors:
- Business, Strategic, Management, Project Management
- Technical, Database, Frontend, Performance, Quality, Testing
- Process, Communication, Team Dynamics, Delivery
- And more...

## Usage

```bash
# From project root
python scripts/create_visualization.py
```

## Dependencies

- **PyYAML**: For parsing YAML frontmatter in markdown files
- **Standard Library**: `os`, `glob`, `json`, `re` modules
- **Frontend**: D3.js (loaded via CDN in the generated HTML)

## Graph Algorithm

The script creates directed edges based on markdown link analysis:

1. **Symptoms Section**: If problem A lists problem B as a symptom, creates edge B → A (B causes A)
2. **Root Causes Section**: If problem A lists problem B as a root cause, creates edge A → B (A is caused by B)
3. **Node Importance**: Calculates incoming connection count to size nodes appropriately
4. **Deduplication**: Removes duplicate edges between the same node pairs

This creates a meaningful causal network where arrows point from causes to effects.
