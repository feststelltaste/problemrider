#!/usr/bin/env python3

import os
import re
import glob

def find_unlinked_content():
    """Find root causes and symptoms that don't contain links in problem files."""
    problems_dir = os.path.join(os.getcwd(), '_problems')
    
    if not os.path.isdir(problems_dir):
        print(f"Error: Directory '{problems_dir}' not found.")
        return

    unlinked_items = []
    
    for filepath in glob.glob(os.path.join(problems_dir, '*.md')):
        filename = os.path.basename(filepath)
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Find Causes and Symptoms sections
        sections = re.finditer(r'^## (Causes ▼|Symptoms ▲)\s*$', content, re.MULTILINE)
        
        for section_match in sections:
            section_type = section_match.group(1)
            section_start = section_match.end()
            
            # Find the next section (or end of file)
            next_section = re.search(r'^## ', content[section_start:], re.MULTILINE)
            if next_section:
                section_end = section_start + next_section.start()
                section_content = content[section_start:section_end]
            else:
                section_content = content[section_start:]
            
            # Find all list items (both - and * bullets)
            list_items = re.findall(r'^[\s]*[-*]\s+(.+?)(?=\n[\s]*[-*]|\n\n|\Z)', 
                                   section_content, re.MULTILINE | re.DOTALL)
            
            for item in list_items:
                # Clean up the item text (remove extra whitespace, newlines)
                item = re.sub(r'\s+', ' ', item.strip())
                
                # Check if the item contains a markdown link
                has_link = bool(re.search(r'\[.+?\]\(.+?\)', item))
                
                if not has_link and item:  # Only include non-empty items without links
                    unlinked_items.append({
                        'file': filename,
                        'section': section_type,
                        'content': item
                    })
    
    # Print results grouped by file
    current_file = None
    for item in sorted(unlinked_items, key=lambda x: (x['file'], x['section'])):
        if item['file'] != current_file:
            if current_file is not None:
                print()
            print(f"## {item['file']}")
            current_file = item['file']
        
        section_symbol = "▲" if "Symptoms" in item['section'] else "▼"
        print(f"  {section_symbol} {item['content']}")
    
    print(f"\nTotal unlinked items found: {len(unlinked_items)}")

if __name__ == '__main__':
    find_unlinked_content()