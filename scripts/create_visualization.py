import os
import glob
import json
import re
import yaml

def parse_frontmatter_and_content(file_content):
    """Parses YAML frontmatter and markdown content from a file string."""
    try:
        parts = file_content.split('---')
        if len(parts) >= 3:
            frontmatter = yaml.safe_load(parts[1])
            content = '---'.join(parts[2:])
            return frontmatter, content
        return None, file_content
    except (yaml.YAMLError, IndexError) as e:
        print(f"Error parsing file: {e}")
        return None, file_content

def create_visualization():
    """
    Generates an HTML/D3.js visualization of the problems and their relationships.
    """
    # Get the project root directory (parent of scripts directory)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    # Find all markdown files in the _problems directory
    pattern = '_problems/*.md'
    problem_files = glob.glob(pathname=pattern, root_dir=project_root)
    
    nodes = []
    edges = []
    title_to_filename = {}
    node_ids = set()

    # First pass: Collect all nodes and map titles to filenames
    for filepath in problem_files:
        filename = os.path.basename(filepath)
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        frontmatter, _ = parse_frontmatter_and_content(content)
        
        if frontmatter and 'title' in frontmatter:
            node_id = filename
            if node_id not in node_ids:
                category = "Uncategorized"
                fm_category = frontmatter.get('category')
                if isinstance(fm_category, list) and fm_category:
                    category = fm_category[0]
                elif isinstance(fm_category, str):
                    category = fm_category

                nodes.append({
                    "id": node_id,
                    "title": frontmatter.get('title', 'No Title'),
                    "description": frontmatter.get('description', 'No Description.'),
                    "category": category
                })
                node_ids.add(node_id)
                title_to_filename[frontmatter['title'].lower()] = node_id

    # Second pass: Create edges and track incoming connections
    incoming_connections = {node['id']: 0 for node in nodes}
    for filepath in problem_files:
        source_filename = os.path.basename(filepath)
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        frontmatter, md_content = parse_frontmatter_and_content(content)
        if not frontmatter:
            continue

        # Split content into sections
        sections = re.split(r'##\s*(Symptoms|Causes)', md_content)
        
        # Process sections to create directed edges
        for i in range(1, len(sections), 2):
            section_title = sections[i]
            section_content = sections[i+1]
            linked_files = re.findall(r'\[([^\]]+)\]\(([^)]+\.md)\)', section_content)

            for _, link_filename in linked_files:
                target_filename = os.path.basename(link_filename)
                if target_filename in node_ids:
                    if section_title == 'Symptoms':
                        # The linked symptom is caused by the current problem
                        # Arrow from the linked symptom to the current problem
                        edges.append({"source": target_filename, "target": source_filename})
                        incoming_connections[source_filename] += 1
                    elif section_title == 'Causes':
                        # The current problem is caused by the linked root cause
                        # Arrow from the current problem to the linked root cause
                        edges.append({"source": source_filename, "target": target_filename})
                        incoming_connections[target_filename] += 1

    # Remove duplicate edges
    unique_edges = [dict(t) for t in {tuple(d.items()) for d in edges}]
    
    # Sort edges alphabetically by source then target for stable generation
    unique_edges.sort(key=lambda x: (x['source'], x['target']))
    
    # Adjust node sizes based on incoming connections (max 20, default 10)
    for node in nodes:
        node['size'] = min(20, 10 + incoming_connections.get(node['id'], 0))
    
    graph_data = {"nodes": nodes, "links": unique_edges}

    # Load the JavaScript template
    js_template_file = os.path.join(script_dir, 'templates', 'visualization.template.js')
    with open(js_template_file, 'r', encoding='utf-8') as f:
        js_template = f.read()
    
    # Replace the placeholder with actual graph data
    js_content = js_template.replace('{GRAPH_DATA}', json.dumps(graph_data, indent=2))

    # Load the Jekyll HTML template
    template_file = os.path.join(script_dir, 'templates', 'landscape.template.html')
    with open(template_file, 'r', encoding='utf-8') as f:
        jekyll_page_content = f.read()

    # Write the JavaScript file
    js_output_file = os.path.join(project_root, 'assets', 'js', 'visualization.js')
    os.makedirs(os.path.dirname(js_output_file), exist_ok=True)
    with open(js_output_file, 'w', encoding='utf-8') as f:
        f.write(js_content)

    # Write the Jekyll page
    page_output_file = os.path.join(project_root, 'landscape.html')
    with open(page_output_file, 'w', encoding='utf-8') as f:
        f.write(jekyll_page_content)

    print(f"Jekyll page '{page_output_file}' created successfully.")
    print(f"Visualization JavaScript '{js_output_file}' created successfully.")

if __name__ == '__main__':
    create_visualization()
