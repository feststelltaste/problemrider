#!/usr/bin/env python3
"""
Causal relationship validator using SD-SCM inspired approach.
Install dependencies: pip install -r scripts/requirements.txt
Additional dependencies: pip install networkx openai
"""

import yaml
import argparse
import re
import json
from pathlib import Path
from typing import Dict, List, Tuple, Set
import networkx as nx

try:
    import numpy as np
except ImportError:
    print("❌ numpy not installed")
    print("Run: pip install numpy")
    exit(1)

try:
    from tqdm import tqdm
except ImportError:
    print("⚠️  tqdm not installed - progress bars will be disabled")
    print("To install: pip install tqdm")

try:
    import markdown
    from bs4 import BeautifulSoup
    from markdown.extensions import toc
    from markdown.treeprocessors import Treeprocessor
    from markdown.extensions import Extension
except ImportError:
    print("❌ markdown/beautifulsoup4 not installed")
    print("Run: pip install -r scripts/requirements.txt")
    exit(1)

try:
    import openai
    import os
except ImportError:
    print("❌ openai not installed")
    print("Run: pip install openai")
    exit(1)


class MarkdownStructureExtractor:
    """Extract structured data from markdown using the markdown library instead of regex."""
    
    def __init__(self):
        self.md = markdown.Markdown(extensions=['toc'])
    
    def parse_markdown_sections(self, content: str) -> Dict[str, str]:
        """Parse markdown content and extract sections by headings."""
        # Convert to HTML with TOC to get structure
        html = self.md.convert(content)
        soup = BeautifulSoup(html, 'html.parser')
        
        sections = {}
        current_section = None
        current_content = []
        
        # Find all elements in order
        for element in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'ul', 'ol', 'blockquote']):
            if element.name.startswith('h'):
                # Save previous section if exists
                if current_section:
                    sections[current_section] = ' '.join(current_content).strip()
                    current_content = []
                
                # Start new section
                current_section = element.get_text().strip()
            elif current_section:
                # Add content to current section
                current_content.append(element.get_text())
        
        # Save last section
        if current_section and current_content:
            sections[current_section] = ' '.join(current_content).strip()
        
        # Reset markdown instance for next use
        self.md.reset()
        return sections
    
    def extract_links_from_section(self, section_html: str) -> List[Dict[str, str]]:
        """Extract markdown links with their context from a section."""
        soup = BeautifulSoup(section_html, 'html.parser')
        links = []
        
        # Find all links
        for link in soup.find_all('a'):
            href = link.get('href', '')
            if href.endswith('.md'):
                # Get the surrounding context (parent element text)
                parent = link.parent
                context = parent.get_text() if parent else link.get_text()
                
                # Clean up the slug
                slug = href.replace('.md', '').replace('../', '').replace('./', '')
                
                # Extract confidence score if present in text after link
                confidence_match = re.search(r'\(([0-9.]+)\)', context)
                confidence = float(confidence_match.group(1)) if confidence_match else None
                
                # Extract description (text after colon)
                description_match = re.search(rf'{re.escape(link.get_text())}\].*?:\s*([^.]*\.?)', context)
                description = description_match.group(1).strip() if description_match else ""
                
                links.append({
                    'title': link.get_text(),
                    'slug': slug,
                    'href': href,
                    'confidence': confidence,
                    'description': description,
                    'full_context': context.strip()
                })
        
        return links


class CausalRelationshipValidator:
    def __init__(self, problems_dir: str = "_problems", skip_cycle_check: bool = False, use_local: bool = False, local_url: str = None):
        self.problems_dir = Path(problems_dir)
        self.problems: Dict[str, Dict] = {}
        self.causal_graph = nx.DiGraph()
        self.skip_cycle_check = skip_cycle_check
        self.use_local = use_local
        self.local_url = local_url or "http://host.docker.internal:1234"
        self.local_model_name = None
        
        # Initialize markdown extractor
        self.md_extractor = MarkdownStructureExtractor()
        
        # Cache for validation results to avoid re-validating same relationships
        self.validation_cache: Dict[str, Dict] = {}
        self.cache_file = Path("scripts/tmp/causal_validation_cache.json")
        
        # Ensure cache directory exists
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        self._load_cache()
        
        # Initialize OpenRouter client for better cost control
        try:
            self.client = openai.OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.getenv("OPENROUTER_API_KEY")
            )
            if not os.getenv("OPENROUTER_API_KEY"):
                print("❌ OPENROUTER_API_KEY environment variable not set")
                print("Set it with: export OPENROUTER_API_KEY='your-key-here'")
                print("Get your key from: https://openrouter.ai/")
                exit(1)
        except Exception as e:
            print(f"❌ Failed to initialize OpenRouter client: {e}")
            exit(1)
    
    def _load_cache(self) -> None:
        """Load validation cache from disk."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    self.validation_cache = json.load(f)
                print(f"📁 Loaded {len(self.validation_cache)} cached validations")
            except Exception as e:
                print(f"⚠️  Could not load cache: {e}")
                self.validation_cache = {}
        else:
            self.validation_cache = {}
    
    def _save_cache(self) -> None:
        """Save validation cache to disk."""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.validation_cache, f, indent=2)
        except Exception as e:
            print(f"⚠️  Could not save cache: {e}")
    
    def _get_cache_key(self, cause: str, effect: str) -> str:
        """Generate cache key for a causal relationship."""
        return f"{cause}->{effect}"
    
    def _get_timestamp(self) -> str:
        """Get current timestamp for validation tracking."""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def _extract_context_specific_description(self, cause: str, effect: str) -> str:
        """Extract context-specific description for causal relationship from existing markdown."""
        # Check if this relationship is already stored in the causal graph with context
        if self.causal_graph.has_edge(cause, effect):
            edge_data = self.causal_graph.edges[cause, effect]
            context = edge_data.get('context', '')
            if context:
                return context
        
        # Fall back to extracting from markdown content
        effect_data = self.problems.get(effect, {})
        markdown_content = effect_data.get('markdown_content', '')
        
        if markdown_content:
            # Convert to HTML and look for the cause link
            html = self.md_extractor.md.convert(markdown_content)
            soup = BeautifulSoup(html, 'html.parser')
            
            # Find links to the cause problem
            for link in soup.find_all('a'):
                href = link.get('href', '')
                if cause in href:
                    # Get the context from parent element
                    parent = link.parent
                    if parent:
                        text = parent.get_text()
                        # Look for description after colon
                        if ':' in text:
                            description = text.split(':', 1)[1].strip()
                            if description and len(description) > 10:  # Meaningful description
                                return description[:200]  # Limit length
            
            # Reset markdown instance
            self.md_extractor.md.reset()
        
        return ""
    
    def load_problems(self) -> None:
        """Load all problem files and extract causal relationships."""
        print(f"📁 Loading problems from {self.problems_dir}...")
        
        problem_files = list(self.problems_dir.glob("*.md"))
        print(f"📄 Found {len(problem_files)} problem files")
        
        try:
            from tqdm import tqdm
            use_tqdm = True
        except ImportError:
            use_tqdm = False
            print("⚠️  tqdm not available, loading without progress bar")
        
        if use_tqdm:
            file_iter = tqdm(problem_files, desc="📖 Loading problems", unit="file")
        else:
            file_iter = problem_files
        
        loaded_count = 0
        error_count = 0
        
        for file_path in file_iter:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                if content.startswith('---'):
                    parts = content.split('---', 2)
                    if len(parts) >= 3:
                        metadata = yaml.safe_load(parts[1])
                        markdown_content = parts[2]
                        
                        problem_key = file_path.stem
                        
                        # Use markdown extractor to parse sections
                        sections = self.md_extractor.parse_markdown_sections(markdown_content)
                        
                        self.problems[problem_key] = {
                            'file_path': file_path,
                            'title': metadata.get('title', ''),
                            'description': metadata.get('description', ''),
                            'metadata': metadata,
                            'markdown_content': markdown_content,
                            'sections': sections,
                            'symptoms': sections.get('Indicators ⟡', sections.get('Indicators', '')),
                            'root_causes': sections.get('Causes ▼', sections.get('Root Causes', sections.get('Causes', ''))),
                            'description_section': sections.get('Description', '')
                        }
                        loaded_count += 1
                        
                        if use_tqdm:
                            file_iter.set_postfix({'loaded': loaded_count, 'errors': error_count})
                        
            except Exception as e:
                error_count += 1
                if not use_tqdm:
                    print(f"❌ Error loading {file_path}: {e}")
                
        print(f"✅ Successfully loaded {loaded_count} problems")
        if error_count > 0:
            print(f"⚠️  {error_count} files had errors during loading")
    
    
    def build_causal_graph(self) -> None:
        """Build initial causal graph from existing problem relationships."""
        print("🕸️  Building causal graph from problem relationships...")
        
        try:
            from tqdm import tqdm
            use_tqdm = True
        except ImportError:
            use_tqdm = False
        
        if use_tqdm:
            problems_iter = tqdm(self.problems.items(), desc="🔗 Extracting relationships", unit="problem")
        else:
            problems_iter = self.problems.items()
            print(f"Processing {len(self.problems)} problems...")
        
        nodes_added = 0
        edges_added = 0
        
        for problem_key, problem_data in problems_iter:
            # Add problem node
            self.causal_graph.add_node(problem_key, 
                                     type='problem',
                                     title=problem_data['title'],
                                     description=problem_data['description'])
            nodes_added += 1
            
            # Extract causal relationships from text
            initial_edge_count = self.causal_graph.number_of_edges()
            self._extract_causal_links(problem_key, problem_data)
            new_edges = self.causal_graph.number_of_edges() - initial_edge_count
            edges_added += new_edges
            
            if use_tqdm:
                problems_iter.set_postfix({'nodes': nodes_added, 'edges': edges_added})
        
        print(f"📊 Causal graph built: {nodes_added} nodes, {edges_added} relationships")
        
        # Check for cycles (optional)
        if not self.skip_cycle_check:
            print("🔄 Checking for circular relationships...")
            self._detect_and_report_cycles()
        else:
            print("⏩ Skipping circular relationship check")
    
    def _extract_causal_links(self, problem_key: str, problem_data: Dict) -> None:
        """Extract causal relationships mentioned in problem text using markdown parsing."""
        markdown_content = problem_data.get('markdown_content', '')
        
        # Convert to HTML to extract links properly
        html = self.md_extractor.md.convert(markdown_content)
        soup = BeautifulSoup(html, 'html.parser')
        
        # Find section headings and their content
        current_section = None
        section_content = {}
        current_html = []
        
        for element in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'ul', 'ol']):
            if element.name.startswith('h'):
                # Save previous section
                if current_section and current_html:
                    section_content[current_section] = ''.join(str(e) for e in current_html)
                    current_html = []
                current_section = element.get_text().strip()
            else:
                current_html.append(element)
        
        # Save last section
        if current_section and current_html:
            section_content[current_section] = ''.join(str(e) for e in current_html)
        
        # Extract links from symptoms sections
        symptoms_sections = ['Indicators ⟡', 'Indicators', 'Symptoms ▲', 'Symptoms']
        for section_name in symptoms_sections:
            if section_name in section_content:
                links = self.md_extractor.extract_links_from_section(section_content[section_name])
                for link in links:
                    if link['slug'] in self.problems:
                        self.causal_graph.add_edge(problem_key, link['slug'], 
                                                 relationship='manifests_as',
                                                 context=link['description'])
        
        # Extract links from root causes sections  
        causes_sections = ['Causes ▼', 'Causes', 'Root Causes']
        for section_name in causes_sections:
            if section_name in section_content:
                links = self.md_extractor.extract_links_from_section(section_content[section_name])
                for link in links:
                    if link['slug'] in self.problems:
                        self.causal_graph.add_edge(link['slug'], problem_key, 
                                                 relationship='causes',
                                                 context=link['description'])
        
        # Reset markdown instance
        self.md_extractor.md.reset()
    
    def _detect_and_report_cycles(self) -> None:
        """Detect and report circular causal relationships."""
        try:
            cycles = list(nx.simple_cycles(self.causal_graph))
            if cycles:
                print(f"⚠️  Found {len(cycles)} circular causal relationships:")
                for i, cycle in enumerate(cycles[:5]):  # Show first 5 cycles
                    cycle_titles = [self.problems[node]['title'] for node in cycle]
                    print(f"  {i+1}. {' → '.join(cycle_titles)} → {cycle_titles[0]}")
                if len(cycles) > 5:
                    print(f"  ... and {len(cycles) - 5} more cycles")
                print("  These may indicate conceptual issues or need validation.")
            else:
                print("✓ No circular causal relationships detected")
        except Exception as e:
            print(f"⚠️  Could not check for cycles: {e}")
    
    def validate_causal_relationship(self, cause: str, effect: str) -> Dict:
        """Use LLM to validate a causal relationship with focus on causal chains."""
        # Check cache first (unless revisit mode is enabled)
        cache_key = self._get_cache_key(cause, effect)
        if cache_key in self.validation_cache and not getattr(self, '_revisit_mode', False):
            print(f"  💾 Using cached result")
            return self.validation_cache[cache_key]
        elif cache_key in self.validation_cache and getattr(self, '_revisit_mode', False):
            print(f"  🔄 Re-validating (revisit mode)")
        else:
            print(f"  🆕 New validation")
        
        cause_info = self.problems.get(cause, {})
        effect_info = self.problems.get(effect, {})
        
        prompt = f"""
You are an expert in legacy software systems and causal reasoning following SD-SCM (Structural Causal Model) principles. Evaluate whether there is a DIRECT CAUSAL CHAIN between these two problems in legacy systems.

POTENTIAL ROOT CAUSE:
Title: {cause_info.get('title', 'Unknown')}
Description: {cause_info.get('description', 'No description')}
Root Causes: {cause_info.get('root_causes', 'None listed')}

POTENTIAL EFFECT/SYMPTOM:
Title: {effect_info.get('title', 'Unknown')}
Description: {effect_info.get('description', 'No description')}
Symptoms: {effect_info.get('symptoms', 'None listed')}

CAUSAL EVALUATION FRAMEWORK:
Apply these tests systematically:

1. TEMPORAL PRECEDENCE: Does the ROOT CAUSE logically precede the EFFECT in legacy system degradation?

2. MECHANISM: What is the specific technical mechanism by which the ROOT CAUSE would produce the EFFECT?
   - Architectural degradation patterns
   - Technical debt accumulation
   - System complexity cascade effects
   - Resource constraint propagation
   - Knowledge loss impacts

3. COUNTERFACTUAL: If we could eliminate the ROOT CAUSE, would the EFFECT be prevented or significantly reduced?

4. ALTERNATIVE EXPLANATIONS: Are there more plausible direct causes for the EFFECT that don't involve the ROOT CAUSE?

5. LEGACY SYSTEM CONTEXT: Does this causal relationship make sense in the context of aging software systems, technical debt, and organizational constraints?

Please respond with a JSON object containing:
- "plausible": boolean - Is there a direct causal relationship?
- "confidence": number (0.0 to 1.0) - Based on strength of evidence and mechanism clarity
- "reasoning": string - Detailed explanation of the causal mechanism or why it's implausible
- "causal_chain": string - Step-by-step process from cause to effect (if plausible)
- "relationship_type": string - One of: "direct_cause", "indirect_cause", "symptom_of", "correlated_only", "unrelated"
- "temporal_order": string - "cause_before_effect", "simultaneous", "unclear", or "effect_before_cause"

Be rigorous: correlation and co-occurrence are NOT causation. Require clear mechanisms and logical necessity.
"""
        
        try:
            response = self.client.chat.completions.create(
                model="openai/gpt-5-nano",
                messages=[
                    {"role": "system", "content": "You are an expert in legacy software systems and causal reasoning."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Parse JSON response
            result = json.loads(response.choices[0].message.content)
            
            # Enhance result with additional context
            enhanced_result = {
                **result,
                'cause_title': cause_info.get('title', 'Unknown'),
                'cause_description': cause_info.get('description', ''),
                'effect_title': effect_info.get('title', 'Unknown'),
                'effect_description': effect_info.get('description', ''),
                'context_specific_explanation': self._extract_context_specific_description(cause, effect),
                'validation_timestamp': self._get_timestamp()
            }
            
            # Cache the enhanced result
            self.validation_cache[cache_key] = enhanced_result
            self._save_cache()
            
            return enhanced_result
            
        except Exception as e:
            print(f"Error validating relationship {cause} -> {effect}: {e}")
            error_result = {"plausible": False, "confidence": 0.0, "reasoning": f"Error: {e}", "alternative_relationship": ""}
            # Cache error results too to avoid retrying immediately
            self.validation_cache[cache_key] = error_result
            self._save_cache()
            return error_result
    
    def validate_all_relationships(self, min_confidence: float = 0.7, limit: int = None, random_start: bool = False, specific_file: str = None, revisit: bool = False) -> List[Dict]:
        """Validate causal relationships in the graph."""
        print(f"🔍 Validating causal relationships with LLM{f' (limit: {limit})' if limit else ''}...")
        
        # Set revisit mode for cache behavior
        self._revisit_mode = revisit
        if revisit:
            print("🔄 Revisit mode enabled: re-validating cached relationships")
        
        validation_results = []
        edges = list(self.causal_graph.edges(data=True))
        total_edges = len(edges)
        
        print(f"📊 Found {total_edges} causal relationships to validate")
        
        # Filter by specific file if requested
        if specific_file:
            # Handle .md extension - remove it to get the slug
            file_slug = specific_file
            if file_slug.endswith('.md'):
                file_slug = file_slug[:-3]
            
            if file_slug not in self.problems:
                print(f"❌ Problem file '{specific_file}' not found. Available files:")
                available_files = list(self.problems.keys())[:10]  # Show first 10
                for f in available_files:
                    print(f"  - {f}.md")
                if len(self.problems) > 10:
                    print(f"  ... and {len(self.problems) - 10} more")
                return []
            
            specific_file = file_slug  # Use slug for rest of the function
            
            # Filter edges to only include relationships involving the specific file
            filtered_edges = [
                edge for edge in edges 
                if edge[0] == specific_file or edge[1] == specific_file
            ]
            edges = filtered_edges
            print(f"📁 Focusing on relationships for '{specific_file}' ({len(edges)} relationships)")
        
        # Randomize order if requested
        if random_start:
            import random
            random.shuffle(edges)
            print("🎲 Starting with random relationships")
        
        # Limit the number of relationships to validate
        if limit:
            edges = edges[:limit]
            selection_method = "random" if random_start else "first"
            if specific_file:
                print(f"⚡ Processing {selection_method} {len(edges)} relationships for '{specific_file}' out of {total_edges} total")
            else:
                print(f"⚡ Processing {selection_method} {len(edges)} relationships out of {total_edges}")
        
        # Import tqdm for progress bar
        try:
            from tqdm import tqdm
            use_tqdm = True
        except ImportError:
            use_tqdm = False
            print("⚠️  tqdm not available, using simple progress indicator")
        
        if use_tqdm:
            edges_iter = tqdm(edges, desc="🧠 Validating relationships", unit="relationship")
        else:
            edges_iter = edges
        
        plausible_count = 0
        implausible_count = 0
        cached_count = 0
        new_count = 0
        
        for i, edge in enumerate(edges_iter, 1):
            cause, effect, data = edge
            relationship_type = data.get('relationship', 'unknown')
            
            # Check if validation is cached
            cache_key = self._get_cache_key(cause, effect)
            is_cached = cache_key in self.validation_cache and not revisit
            
            if is_cached:
                cached_count += 1
            else:
                new_count += 1
            
            if use_tqdm:
                edges_iter.set_postfix({
                    'plausible': plausible_count,
                    'implausible': implausible_count,
                    'cached': cached_count,
                    'new': new_count
                })
            else:
                # Simple progress for non-tqdm case
                progress = f"[{i}/{len(edges)}]"
                print(f"{progress} Validating: {cause} --{relationship_type}--> {effect}")
            
            validation = self.validate_causal_relationship(cause, effect)
            
            result = {
                'cause': cause,
                'effect': effect,
                'relationship_type': relationship_type,
                'validation': validation,
                'needs_review': validation['confidence'] < min_confidence or not validation['plausible']
            }
            
            # Update counters
            if validation['plausible']:
                plausible_count += 1
            else:
                implausible_count += 1
            
            # Display result for non-tqdm case
            if not use_tqdm:
                status = "✓" if validation['plausible'] else "✗"
                print(f"  {status} Result: {validation['plausible']} (confidence: {validation['confidence']:.2f})")
                print(f"  Reasoning: {validation['reasoning'][:100]}...")
                print(f"  Progress: {i}/{len(edges)} ({i/len(edges)*100:.1f}%)")
                print()
            
            validation_results.append(result)
            
            # Add to graph for analysis
            self.causal_graph.edges[cause, effect]['validation'] = validation
        
        print(f"\n📊 Validation Summary:")
        print(f"  ✅ Plausible relationships: {plausible_count}")
        print(f"  ❌ Implausible relationships: {implausible_count}")
        print(f"  💾 Used cached results: {cached_count}")
        print(f"  🆕 New validations: {new_count}")
        
        return validation_results
    
    def find_missing_causal_relationships(self, sample_size: int = 10) -> List[Dict]:
        """Find potential missing causal relationships using both semantic similarity and exhaustive search."""
        print(f"🔍 Looking for missing causal relationships (sample size: {sample_size})...")
        
        missing_relationships = []
        
        # Get problems that have symptoms but unclear causes
        problems_needing_causes = [
            key for key, data in self.problems.items() 
            if data['symptoms'] and not data['root_causes']
        ]
        
        print(f"📝 Found {len(problems_needing_causes)} problems that need cause analysis")
        
        # Sample problems to analyze
        import random
        from tqdm import tqdm
        
        sample_problems = random.sample(
            problems_needing_causes, 
            min(sample_size, len(problems_needing_causes))
        )
        
        print(f"🎲 Selected {len(sample_problems)} problems for analysis: {', '.join(sample_problems)}")
        
        for i, problem_key in enumerate(sample_problems, 1):
            print(f"\n[{i}/{len(sample_problems)}] 🔍 Analyzing '{problem_key}' for missing causes...")
            
            # Strategy 1: Use semantic similarity for quick relevant problem discovery
            print("  📊 Finding semantically similar problems...")
            relevant_candidates = self._find_semantically_similar_problems(problem_key, top_k=10)
            
            # Strategy 2: Also include random sample for non-obvious causal relationships
            all_other_problems = [p for p in self.problems.keys() if p != problem_key]
            random_candidates = random.sample(all_other_problems, min(5, len(all_other_problems)))
            
            # Combine candidates (remove duplicates)
            all_candidates = list(set(relevant_candidates + random_candidates))
            
            print(f"  🧪 Testing {len(all_candidates)} potential causes...")
            
            with tqdm(all_candidates, desc=f"  Testing causes for {problem_key}", leave=False) as pbar:
                for potential_cause in pbar:
                    pbar.set_postfix_str(f"Testing {potential_cause[:20]}...")
                    
                    # Check if relationship already exists
                    if not self.causal_graph.has_edge(potential_cause, problem_key):
                        validation = self.validate_causal_relationship(potential_cause, problem_key)
                        
                        if validation['plausible'] and validation['confidence'] > 0.6:
                            print(f"\n    ✅ FOUND: {potential_cause} -> {problem_key} (confidence: {validation['confidence']:.2f})")
                            print(f"       Reasoning: {validation['reasoning'][:100]}...")
                            missing_relationships.append({
                                'cause': potential_cause,
                                'effect': problem_key,
                                'validation': validation,
                                'suggested': True,
                                'discovery_method': 'semantic' if potential_cause in relevant_candidates else 'random'
                            })
        
        print(f"\n🎯 Found {len(missing_relationships)} new causal relationships!")
        return missing_relationships
    
    def generate_symptoms_from_causes(self, problem_key: str, validated_causes: List[Dict]) -> Dict:
        """Generate symptoms for a problem based on its validated root causes."""
        problem_data = self.problems.get(problem_key, {})
        
        # Prepare cause information for the LLM
        cause_descriptions = []
        for cause_info in validated_causes:
            cause_key = cause_info['cause']
            cause_data = self.problems.get(cause_key, {})
            confidence = cause_info.get('confidence', 0.0)
            
            cause_descriptions.append({
                'title': cause_data.get('title', 'Unknown'),
                'description': cause_data.get('description', ''),
                'confidence': confidence,
                'reasoning': cause_info.get('reasoning', '')
            })
        
        prompt = f"""
You are an expert in legacy software systems analysis. Based on the validated root causes below, generate comprehensive and observable symptoms for this problem.

PROBLEM TO ANALYZE:
Title: {problem_data.get('title', 'Unknown')}
Description: {problem_data.get('description', 'No description')}

VALIDATED ROOT CAUSES:
{self._format_causes_for_prompt(cause_descriptions)}

TASK: Generate observable symptoms that would manifest when this problem occurs, given these specific root causes.

REQUIREMENTS:
1. Focus on OBSERVABLE and MEASURABLE symptoms
2. Each symptom should logically derive from the root causes
3. Include both technical and organizational symptoms where applicable
4. Prioritize symptoms that are early warning indicators
5. Make symptoms specific enough to be actionable for detection

Please respond with a JSON object containing:
- "symptoms": array of symptom objects, each with:
  - "description": string - Observable symptom description
  - "type": string - One of: "technical", "organizational", "behavioral", "performance", "quality"
  - "detectability": string - One of: "early", "moderate", "late" 
  - "measurement": string - How this symptom can be measured or observed
  - "related_causes": array - Which root cause titles this symptom stems from
- "confidence": number (0.0 to 1.0) - Overall confidence in symptom set
- "reasoning": string - Why these symptoms logically follow from the causes
"""
        
        try:
            response = self.client.chat.completions.create(
                model="openai/gpt-5-nano",
                messages=[
                    {"role": "system", "content": "You are an expert in legacy software systems and symptom analysis."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            result = json.loads(response.choices[0].message.content)
            
            # Enhance with metadata
            enhanced_result = {
                **result,
                'problem_key': problem_key,
                'problem_title': problem_data.get('title', 'Unknown'),
                'generation_timestamp': self._get_timestamp(),
                'source_causes': [c['title'] for c in cause_descriptions]
            }
            
            return enhanced_result
            
        except Exception as e:
            print(f"Error generating symptoms for {problem_key}: {e}")
            return {
                "symptoms": [],
                "confidence": 0.0,
                "reasoning": f"Error: {e}",
                "problem_key": problem_key,
                "generation_timestamp": self._get_timestamp()
            }
    
    def _format_causes_for_prompt(self, cause_descriptions: List[Dict]) -> str:
        """Format cause descriptions for the LLM prompt."""
        formatted = []
        for i, cause in enumerate(cause_descriptions, 1):
            formatted.append(
                f"{i}. **{cause['title']}** (Confidence: {cause['confidence']:.2f})\n"
                f"   Description: {cause['description']}\n"
                f"   Validation Reasoning: {cause['reasoning'][:200]}..."
            )
        return "\n\n".join(formatted)
    
    def generate_symptoms_for_validated_problems(self, validation_results: List[Dict], min_confidence: float = 0.7) -> Dict[str, Dict]:
        """Generate symptoms for all problems that have validated causes."""
        print(f"Generating symptoms from validated causes (min confidence: {min_confidence})...")
        
        # Group validated causes by effect (problem)
        causes_by_problem = {}
        for result in validation_results:
            if result['validation']['plausible'] and result['validation']['confidence'] >= min_confidence:
                effect = result['effect']
                if effect not in causes_by_problem:
                    causes_by_problem[effect] = []
                
                causes_by_problem[effect].append({
                    'cause': result['cause'],
                    'confidence': result['validation']['confidence'],
                    'reasoning': result['validation'].get('reasoning', ''),
                    'relationship_type': result['validation'].get('relationship_type', 'causes')
                })
        
        # Generate symptoms for each problem with validated causes
        generated_symptoms = {}
        for problem_key, causes in causes_by_problem.items():
            if len(causes) >= 1:  # Need at least 1 validated cause
                print(f"🧬 Generating symptoms for '{problem_key}' from {len(causes)} validated causes...")
                
                symptoms = self.generate_symptoms_from_causes(problem_key, causes)
                generated_symptoms[problem_key] = symptoms
                
                # Display summary
                if 'symptoms' in symptoms and symptoms['symptoms']:
                    print(f"  ✓ Generated {len(symptoms['symptoms'])} symptoms (confidence: {symptoms.get('confidence', 0):.2f})")
                    for symptom in symptoms['symptoms'][:2]:  # Show first 2
                        print(f"    - {symptom.get('description', '')[:80]}...")
                else:
                    print(f"  ⚠️ No symptoms generated")
                print()
        
        return generated_symptoms
    
    def _get_content_hash(self, problem_data: Dict) -> str:
        """Generate MD5 hash of problem content for cache validation."""
        import hashlib
        
        # Combine the content that affects embeddings
        content_parts = [
            problem_data['title'],
            problem_data['description'],
            problem_data.get('description_section', ''),
            problem_data.get('symptoms', '')
        ]
        content_text = " ".join(part for part in content_parts if part.strip())
        
        return hashlib.md5(content_text.encode()).hexdigest()
    
    def _normalize_model_name(self, model_name: str) -> str:
        """Normalize model names to ensure consistency between local and direct loading."""
        # Map local LM Studio model names to standard names
        normalization_map = {
            "text-embedding-qwen3-embedding-0.6b": "Qwen/Qwen3-Embedding-0.6B",
            "text-embedding-qwen3-embedding-4b": "Qwen/Qwen3-Embedding-4B", 
            "text-embedding-qwen3-embedding-8b": "Qwen/Qwen3-Embedding-8B",
            "text-embedding-all-minilm-l6-v2-embedding": "all-MiniLM-L6-v2",
            "text-embedding-nomic-embed-text-v1.5": "nomic-ai/nomic-embed-text-v1.5"
        }
        
        return normalization_map.get(model_name, model_name)
    
    def _setup_local_service(self) -> str:
        """Setup connection to local embedding service and return model name."""
        try:
            import requests
        except ImportError:
            print("❌ requests package required for local embedding service")
            print("Run: pip install requests")
            raise
        
        print(f"🌐 Using local embedding service at: {self.local_url}")
        
        try:
            # Get available models
            response = requests.get(f"{self.local_url}/v1/models", timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data.get('data'):
                    model_names = [model.get('id', 'unknown') for model in data['data']]
                    print(f"✅ LM Studio available with {len(model_names)} models: {', '.join(model_names[:3])}{'...' if len(model_names) > 3 else ''}")
                    
                    # Try to find preferred model
                    preferred_model = "text-embedding-qwen3-embedding-0.6b"
                    if preferred_model in model_names:
                        self.local_model_name = preferred_model
                        print(f"🎯 Using preferred model: {self.local_model_name}")
                        return self._normalize_model_name(preferred_model)
                    else:
                        # Use first available model
                        self.local_model_name = model_names[0]
                        print(f"⚠️  Using first available model: {self.local_model_name}")
                        return self._normalize_model_name(self.local_model_name)
                else:
                    raise ValueError("No models available in local service")
            else:
                raise ValueError(f"Local service responded with status {response.status_code}")
        except Exception as e:
            print(f"❌ Cannot connect to LM Studio: {e}")
            print("🔄 Falling back to direct model loading...")
            self.use_local = False
            from sentence_transformers import SentenceTransformer
            print("🤖 Loading Qwen3-Embedding-0.6B model directly...")
            return "Qwen/Qwen3-Embedding-0.6B"
    
    def _get_embedding_from_local_service(self, text: str) -> np.ndarray:
        """Get embedding from local service."""
        try:
            import requests
            
            response = requests.post(
                f"{self.local_url}/v1/embeddings",
                json={
                    "input": text,
                    "model": self.local_model_name
                },
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            return np.array(data["data"][0]["embedding"])
        except Exception as e:
            print(f"❌ Error getting embedding from local service: {e}")
            raise
    
    def _load_cached_embedding(self, problem_key: str, expected_model: str) -> tuple:
        """Load cached embedding from _embeddings directory if available and valid."""
        embedding_file = Path("_embeddings") / f"{problem_key}.yaml"
        
        if not embedding_file.exists():
            return None, None, "no_cache_file"
        
        try:
            with open(embedding_file, 'r', encoding='utf-8') as f:
                cache_data = yaml.safe_load(f)
            
            cached_embedding = cache_data.get('embedding')
            cached_hash = cache_data.get('hash')
            cached_model = cache_data.get('model')
            
            # Validate all required fields exist
            if not cached_embedding or not cached_hash or not cached_model:
                return None, None, "incomplete_cache_data"
            
            # Normalize both model names for comparison
            normalized_cached = self._normalize_model_name(cached_model)
            normalized_expected = self._normalize_model_name(expected_model)
            
            # Check if the models are equivalent after normalization
            if normalized_cached != normalized_expected:
                return None, None, f"model_mismatch_cached:{normalized_cached}_expected:{normalized_expected}"
            
            return np.array(cached_embedding), cached_hash, "valid_cache"
            
        except Exception as e:
            return None, None, f"cache_load_error:{str(e)}"
    
    def _find_semantically_similar_problems(self, problem_key: str, top_k: int = 10) -> List[str]:
        """Find semantically similar problems using cached embeddings or compute new ones."""
        try:
            import numpy as np
            from tqdm import tqdm
            
            print(f"🔍 Finding semantically similar problems for '{problem_key}'...")
            
            # Initialize model based on use_local setting
            if self.use_local:
                expected_model = self._setup_local_service()
                model = None  # No local model needed for API calls
            else:
                from sentence_transformers import SentenceTransformer
                expected_model = "Qwen/Qwen3-Embedding-0.6B"
                print(f"🤖 Loading {expected_model} model...")
                model = SentenceTransformer(expected_model)
            
            # Get or compute embedding for target problem
            target_problem = self.problems[problem_key]
            target_content_hash = self._get_content_hash(target_problem)
            
            # Try to load cached embedding with validation
            target_embedding, cached_hash, cache_status = self._load_cached_embedding(problem_key, expected_model)
            
            if target_embedding is None or cached_hash != target_content_hash:
                cache_reason = "content_changed" if cached_hash != target_content_hash else cache_status
                print(f"  🆕 Computing new embedding for {problem_key} (reason: {cache_reason})")
                target_text = f"{target_problem['title']} {target_problem['description']} {target_problem['symptoms']}"
                
                if self.use_local:
                    target_embedding = self._get_embedding_from_local_service(target_text)
                else:
                    target_embedding = model.encode([target_text])[0]
            else:
                print(f"  💾 Using cached embedding for {problem_key}")
            
            # Get embeddings for all other problems
            print(f"📊 Calculating similarities with {len(self.problems) - 1} other problems...")
            similarities = []
            cached_count = 0
            new_count = 0
            cache_issues = {"no_cache": 0, "model_mismatch": 0, "content_changed": 0, "errors": 0}
            
            other_problems = [(key, data) for key, data in self.problems.items() if key != problem_key]
            
            with tqdm(other_problems, desc="🧮 Computing similarities", unit="problem") as pbar:
                for other_key, other_data in pbar:
                    # Try to load cached embedding with full validation
                    other_content_hash = self._get_content_hash(other_data)
                    other_embedding, other_cached_hash, other_cache_status = self._load_cached_embedding(other_key, expected_model)
                    
                    use_cache = (other_embedding is not None and 
                                other_cached_hash == other_content_hash and
                                other_cache_status == "valid_cache")
                    
                    if not use_cache:
                        # Compute new embedding
                        other_text = f"{other_data['title']} {other_data['description']} {other_data['root_causes']}"
                        
                        if self.use_local:
                            other_embedding = self._get_embedding_from_local_service(other_text)
                        else:
                            other_embedding = model.encode([other_text])[0]
                        new_count += 1
                        
                        # Track cache issue reasons
                        if "no_cache" in other_cache_status:
                            cache_issues["no_cache"] += 1
                        elif "model_mismatch" in other_cache_status:
                            cache_issues["model_mismatch"] += 1
                        elif other_cached_hash != other_content_hash:
                            cache_issues["content_changed"] += 1
                        else:
                            cache_issues["errors"] += 1
                    else:
                        cached_count += 1
                    
                    pbar.set_postfix({"cached": cached_count, "new": new_count})
                    
                    # Cosine similarity
                    similarity = np.dot(target_embedding, other_embedding) / (
                        np.linalg.norm(target_embedding) * np.linalg.norm(other_embedding)
                    )
                    similarities.append((other_key, similarity))
            
            # Report cache statistics
            print(f"  📁 Used {cached_count} cached embeddings, computed {new_count} new ones")
            if sum(cache_issues.values()) > 0:
                issue_summary = [f"{k}:{v}" for k, v in cache_issues.items() if v > 0]
                print(f"  🔧 Cache issues: {', '.join(issue_summary)}")
            
            # Return top similar problems
            similarities.sort(key=lambda x: x[1], reverse=True)
            top_similar = [key for key, _ in similarities[:top_k]]
            
            print(f"  🎯 Top {len(top_similar)} similar problems: {', '.join(top_similar[:3])}{'...' if len(top_similar) > 3 else ''}")
            return top_similar
            
        except ImportError:
            print("⚠️  sentence-transformers not available, using random selection")
            import random
            all_other = [p for p in self.problems.keys() if p != problem_key]
            return random.sample(all_other, min(top_k, len(all_other)))
    
    def generate_report(self, validation_results: List[Dict], missing_relationships: List[Dict], generated_symptoms: Dict[str, Dict] = None) -> str:
        """Generate a validation report."""
        report = []
        report.append("# Causal Relationship Validation Report\n")
        
        # Summary statistics
        total_relationships = len(validation_results)
        valid_relationships = sum(1 for r in validation_results if r['validation']['plausible'])
        needs_review = sum(1 for r in validation_results if r['needs_review'])
        
        report.append(f"## Summary")
        report.append(f"- Total relationships validated: {total_relationships}")
        report.append(f"- Plausible relationships: {valid_relationships} ({valid_relationships/total_relationships*100:.1f}%)")
        report.append(f"- Relationships needing review: {needs_review}")
        report.append(f"- Missing relationships found: {len(missing_relationships)}")
        if generated_symptoms:
            total_symptoms = sum(len(symptoms.get('symptoms', [])) for symptoms in generated_symptoms.values())
            report.append(f"- Generated symptoms: {total_symptoms} across {len(generated_symptoms)} problems")
        report.append("")
        
        # Problematic relationships
        problematic = [r for r in validation_results if r['needs_review']]
        if problematic:
            report.append("## Relationships Needing Review")
            for rel in problematic:
                cause_title = self.problems[rel['cause']]['title']
                effect_title = self.problems[rel['effect']]['title']
                confidence = rel['validation']['confidence']
                reasoning = rel['validation']['reasoning']
                
                report.append(f"### {cause_title} → {effect_title}")
                report.append(f"**Confidence:** {confidence:.2f}")
                report.append(f"**Reasoning:** {reasoning}")
                report.append("")
        
        # Missing relationships
        if missing_relationships:
            report.append("## Suggested Missing Relationships")
            for rel in missing_relationships:
                cause_title = self.problems[rel['cause']]['title']
                effect_title = self.problems[rel['effect']]['title']
                confidence = rel['validation']['confidence']
                reasoning = rel['validation']['reasoning']
                
                report.append(f"### {cause_title} → {effect_title}")
                report.append(f"**Confidence:** {confidence:.2f}")
                report.append(f"**Reasoning:** {reasoning}")
                report.append("")
        
        # Generated symptoms section
        if generated_symptoms:
            report.append("## Generated Symptoms from Validated Causes")
            for problem_key, symptoms_data in generated_symptoms.items():
                problem_title = symptoms_data.get('problem_title', problem_key)
                confidence = symptoms_data.get('confidence', 0.0)
                reasoning = symptoms_data.get('reasoning', '')
                symptoms = symptoms_data.get('symptoms', [])
                
                report.append(f"### {problem_title}")
                report.append(f"**Generation Confidence:** {confidence:.2f}")
                report.append(f"**Source Causes:** {', '.join(symptoms_data.get('source_causes', []))}")
                report.append(f"**Reasoning:** {reasoning}")
                report.append("")
                
                if symptoms:
                    report.append("**Generated Symptoms:**")
                    for symptom in symptoms:
                        desc = symptom.get('description', 'Unknown')
                        sym_type = symptom.get('type', 'unknown')
                        detectability = symptom.get('detectability', 'unknown')
                        measurement = symptom.get('measurement', 'Unknown')
                        
                        report.append(f"- **{desc}**")
                        report.append(f"  - Type: {sym_type}")
                        report.append(f"  - Detectability: {detectability}")
                        report.append(f"  - Measurement: {measurement}")
                        report.append("")
                else:
                    report.append("*No symptoms generated*")
                report.append("")
        
        return "\n".join(report)
    
    def update_problem_files_with_confidence(self, validation_results: List[Dict], missing_relationships: List[Dict]):
        """Update problem files with confidence scores in links and reorder by confidence."""
        print("Updating problem files with confidence scores...")
        
        # Group relationships by effect (the problem being updated)
        relationships_by_effect = {}
        
        # Add validated existing relationships
        for result in validation_results:
            effect = result['effect']
            if effect not in relationships_by_effect:
                relationships_by_effect[effect] = []
            relationships_by_effect[effect].append({
                'cause': result['cause'],
                'confidence': round(result['validation']['confidence'] * 20) / 20,
                'relationship_type': result['relationship_type']
            })
        
        # Add missing relationships that were found
        for result in missing_relationships:
            effect = result['effect']
            if effect not in relationships_by_effect:
                relationships_by_effect[effect] = []
            relationships_by_effect[effect].append({
                'cause': result['cause'],
                'confidence': round(result['validation']['confidence'] * 20) / 20,
                'relationship_type': 'causes'
            })
        
        # Update each problem file
        for problem_key, relationships in relationships_by_effect.items():
            if problem_key in self.problems:
                # Sort by confidence descending
                relationships.sort(key=lambda x: x['confidence'], reverse=True)
                self._update_file_with_confidence(problem_key, relationships)
    
    def _update_file_with_confidence(self, problem_key: str, relationships: List[Dict]):
        """Update a single file with confidence scores in links."""
        problem_data = self.problems[problem_key]
        file_path = problem_data['file_path']
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Update symptoms and root causes sections with confidence scores
        content = self._update_section_with_confidence(content, relationships, 'symptoms')
        content = self._update_section_with_confidence(content, relationships, 'root_causes')
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
            
        print(f"Updated {problem_key} with {len(relationships)} causal relationships")
    
    def _update_section_with_confidence(self, content: str, relationships: List[Dict], section_type: str) -> str:
        """Update a section with confidence scores and reorder by confidence."""
        section_name = "Indicators" if section_type == 'symptoms' else "Root Causes"
        
        # Find section boundaries
        section_start = re.search(rf'^## {section_name}.*$', content, re.MULTILINE)
        if not section_start:
            return content
        
        # Find next section or end of content
        next_section = re.search(r'^## ', content[section_start.end():], re.MULTILINE)
        if next_section:
            section_end = section_start.end() + next_section.start()
        else:
            section_end = len(content)
        
        section_content = content[section_start.end():section_end]
        
        # Extract and update links with confidence scores
        updated_section = self._update_links_with_confidence(section_content, relationships)
        
        # Reconstruct content
        return content[:section_start.end()] + updated_section + content[section_end:]
    
    def _update_links_with_confidence(self, section_content: str, relationships: List[Dict]) -> str:
        """Update markdown links with confidence scores."""
        # Pattern to match markdown links: [title](problem.md) or [title](problem.md) (0.85)
        link_pattern = r'\[([^\]]+)\]\(([^)]+)\.md\)(?:\s*\([0-9.]+\))?'
        
        def replace_link(match):
            title = match.group(1)
            problem_slug = match.group(2)
            
            # Find confidence for this relationship
            confidence = None
            for rel in relationships:
                if rel['cause'] == problem_slug:
                    confidence = rel['confidence']
                    break
            
            if confidence is not None:
                rounded_confidence = round(confidence * 20) / 20
                return f"[{title}]({problem_slug}.md) ({rounded_confidence:.2f})"
            else:
                return match.group(0)  # Return original if no confidence found
        
        return re.sub(link_pattern, replace_link, section_content)


def main():
    parser = argparse.ArgumentParser(
        description="Validate causal relationships and generate symptoms",
        epilog="""Examples:
  %(prog)s             # Full analysis: validate causes + generate symptoms + update files
  %(prog)s --test-run  # Quick test with 3 relationships to check if working
  %(prog)s --skip-cycles  # Skip circular relationship detection for faster startup
  %(prog)s --use-local # Use local LM Studio embedding service
  %(prog)s --file feature-factory  # Analyze only relationships involving 'feature-factory' problem
  %(prog)s --file feature-factory.md --skip-cycles --use-local  # Combine all options""",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--test-run", action="store_true", help="🧪 Test mode: analyze 3 relationships only to verify script works")
    parser.add_argument("--skip-cycles", action="store_true", help="⏩ Skip checking for circular relationships (faster startup)")
    parser.add_argument("--use-local", action="store_true", help="🌐 Use local embedding service instead of sentence-transformers")
    parser.add_argument("--local-url", type=str, help="URL of local embedding service (default: http://host.docker.internal:1234)")
    parser.add_argument("--file", type=str, help="📁 Process only relationships involving a specific problem file (e.g., 'feature-factory' or 'feature-factory.md')")
    
    args = parser.parse_args()
    
    # Set simple defaults
    limit = 3 if args.test_run else None  # Test with 3, otherwise do all
    specific_file = args.file if hasattr(args, 'file') else None
    
    # Show what we're doing
    print("🔍 CAUSAL RELATIONSHIP ANALYSIS")
    print("=" * 50)
    if args.test_run:
        print("🧪 Test mode: analyzing 3 relationships to verify functionality")
    elif specific_file:
        print(f"📁 Single file mode: analyzing relationships for '{specific_file}'")
    else:
        print("🚀 Full analysis mode")
    print()
    
    validator = CausalRelationshipValidator(
        skip_cycle_check=args.skip_cycles,
        use_local=args.use_local,
        local_url=args.local_url
    )
    validator.load_problems()
    validator.build_causal_graph()
    
    # Auto-generate output filename
    output_file = f"causal_analysis_{validator._get_timestamp().replace(' ', '_').replace(':', '-')}.md"
    
    # Step 1: Validate existing causal relationships
    print("STEP 1: Validating causal relationships...")
    validation_results = validator.validate_all_relationships(
        limit=limit,
        random_start=False,
        min_confidence=0.7,
        specific_file=specific_file
    )
    
    # Step 2: Find missing relationships
    missing_relationships = []
    if not args.test_run:  # Skip for test runs
        print("\nSTEP 2: Finding missing causal relationships...")
        missing_relationships = validator.find_missing_causal_relationships(sample_size=5)
    else:
        print("\nSTEP 2: Skipping missing relationship detection (test mode)")
    
    # Step 3: Generate symptoms from validated causes
    generated_symptoms = {}
    if not args.test_run:  # Skip for test runs
        print("\nSTEP 3: Generating symptoms from validated causes...")
        generated_symptoms = validator.generate_symptoms_for_validated_problems(
            validation_results, 
            min_confidence=0.7
        )
    else:
        print("\nSTEP 3: Skipping symptom generation (test mode)")
    
    # Step 4: Generate comprehensive report
    print(f"\nSTEP 4: Generating analysis report...")
    report = validator.generate_report(validation_results, missing_relationships, generated_symptoms)
    
    # Step 5: Save report
    with open(output_file, 'w') as f:
        f.write(report)
    print(f"📄 Report saved to: {output_file}")
    
    # Step 6: Update files
    if not args.test_run:
        print("\nSTEP 5: Updating markdown files with results...")
        if specific_file:
            # For single file mode, only update the specific file and its relationships
            print(f"📁 Updating only files related to '{specific_file}'...")
        validator.update_problem_files_with_confidence(validation_results, missing_relationships)
        print("✅ Files updated with confidence scores")
    else:
        print("\nSTEP 5: Skipping file updates (test mode)")
    
    # Summary
    mode_description = "Test" if args.test_run else (f"Single file ({specific_file})" if specific_file else "Full analysis")
    print(f"\n🎉 {mode_description} complete!")
    
    if specific_file and not args.test_run:
        print(f"\n📊 Summary for '{specific_file}':")
        file_validations = [r for r in validation_results if r['cause'] == specific_file or r['effect'] == specific_file]
        if file_validations:
            plausible_count = sum(1 for r in file_validations if r['validation']['plausible'])
            print(f"  🔗 {len(file_validations)} relationships analyzed")
            print(f"  ✅ {plausible_count} plausible relationships found")
            print(f"  ❌ {len(file_validations) - plausible_count} implausible relationships")
        else:
            print("  ℹ️ No relationships found for this problem")


if __name__ == "__main__":
    main()