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
    import markdown
    from bs4 import BeautifulSoup
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


class CausalRelationshipValidator:
    def __init__(self, problems_dir: str = "_problems"):
        self.problems_dir = Path(problems_dir)
        self.problems: Dict[str, Dict] = {}
        self.causal_graph = nx.DiGraph()
        
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
    
    def load_problems(self) -> None:
        """Load all problem files and extract causal relationships."""
        print(f"Loading problems from {self.problems_dir}...")
        
        for file_path in self.problems_dir.glob("*.md"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                if content.startswith('---'):
                    parts = content.split('---', 2)
                    if len(parts) >= 3:
                        metadata = yaml.safe_load(parts[1])
                        markdown_content = parts[2]
                        
                        problem_key = file_path.stem
                        self.problems[problem_key] = {
                            'file_path': file_path,
                            'title': metadata.get('title', ''),
                            'description': metadata.get('description', ''),
                            'metadata': metadata,
                            'symptoms': self._extract_section(markdown_content, 'Indicators'),
                            'root_causes': self._extract_section(markdown_content, 'Causes'),
                            'description_section': self._extract_section(markdown_content, 'Description')
                        }
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                
        print(f"Loaded {len(self.problems)} problems")
    
    def _extract_section(self, content: str, section_name: str) -> str:
        """Extract a specific section from markdown content."""
        pattern = rf'## {section_name}.*?\n(.*?)(?=\n## |\n$)'
        match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
        if match:
            return self._clean_markdown_text(match.group(1))
        return ''
    
    def _clean_markdown_text(self, text: str) -> str:
        """Clean markdown formatting from text."""
        html = markdown.markdown(text)
        soup = BeautifulSoup(html, 'html.parser')
        plain_text = soup.get_text()
        return re.sub(r'\s+', ' ', plain_text).strip()
    
    def build_causal_graph(self) -> None:
        """Build initial causal graph from existing problem relationships."""
        print("Building causal graph...")
        
        for problem_key, problem_data in self.problems.items():
            # Add problem node
            self.causal_graph.add_node(problem_key, 
                                     type='problem',
                                     title=problem_data['title'],
                                     description=problem_data['description'])
            
            # Extract causal relationships from text
            self._extract_causal_links(problem_key, problem_data)
        
        # Check for cycles
        self._detect_and_report_cycles()
    
    def _extract_causal_links(self, problem_key: str, problem_data: Dict) -> None:
        """Extract causal relationships mentioned in problem text."""
        # Look for linked problems in symptoms and root causes sections
        symptoms_text = problem_data['symptoms']
        root_causes_text = problem_data['root_causes']
        
        # Enhanced regex patterns to find markdown links
        link_patterns = [
            r'\[([^\]]+)\]\(([^)]+)\.md\)',  # Standard markdown links
            r'\[([^\]]+)\]\(([^)]+)\.md\)\s*\([0-9.]+\)',  # Links with confidence scores
            r'\[([^\]]+)\]\(\.\.?/([^)]+)\.md\)',  # Relative path links
        ]
        
        # Find symptom relationships
        for pattern in link_patterns:
            for match in re.finditer(pattern, symptoms_text):
                linked_problem = match.group(2).replace('../', '').replace('./', '')
                if linked_problem in self.problems:
                    # This problem manifests as symptoms in the linked problem
                    self.causal_graph.add_edge(problem_key, linked_problem, 
                                             relationship='manifests_as')
        
        # Find root cause relationships
        for pattern in link_patterns:
            for match in re.finditer(pattern, root_causes_text):
                linked_problem = match.group(2).replace('../', '').replace('./', '')
                if linked_problem in self.problems:
                    # The linked problem causes this problem
                    self.causal_graph.add_edge(linked_problem, problem_key, 
                                             relationship='causes')
    
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
        # Check cache first
        cache_key = self._get_cache_key(cause, effect)
        if cache_key in self.validation_cache:
            print(f"  💾 Using cached result")
            return self.validation_cache[cache_key]
        
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
            
            # Cache the result
            self.validation_cache[cache_key] = result
            self._save_cache()
            
            return result
            
        except Exception as e:
            print(f"Error validating relationship {cause} -> {effect}: {e}")
            error_result = {"plausible": False, "confidence": 0.0, "reasoning": f"Error: {e}", "alternative_relationship": ""}
            # Cache error results too to avoid retrying immediately
            self.validation_cache[cache_key] = error_result
            self._save_cache()
            return error_result
    
    def validate_all_relationships(self, min_confidence: float = 0.7, limit: int = None, random_start: bool = False, specific_file: str = None) -> List[Dict]:
        """Validate causal relationships in the graph."""
        print(f"Validating causal relationships with LLM{f' (limit: {limit})' if limit else ''}...")
        
        validation_results = []
        edges = list(self.causal_graph.edges(data=True))
        total_edges = len(edges)
        
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
                print(f"Processing {selection_method} {len(edges)} relationships for '{specific_file}' out of {total_edges} total")
            else:
                print(f"Processing {selection_method} {len(edges)} relationships out of {total_edges}")
        
        for i, edge in enumerate(edges, 1):
            cause, effect, data = edge
            relationship_type = data.get('relationship', 'unknown')
            
            # Progress indicator
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
            
            # Display result immediately with progress
            status = "✓" if validation['plausible'] else "✗"
            print(f"  {status} Result: {validation['plausible']} (confidence: {validation['confidence']:.2f})")
            print(f"  Reasoning: {validation['reasoning'][:100]}...")
            print(f"  Progress: {i}/{len(edges)} ({i/len(edges)*100:.1f}%)")
            print()
            
            validation_results.append(result)
            
            # Add to graph for analysis
            self.causal_graph.edges[cause, effect]['validation'] = validation
        
        return validation_results
    
    def find_missing_causal_relationships(self, sample_size: int = 10) -> List[Dict]:
        """Find potential missing causal relationships using both semantic similarity and exhaustive search."""
        print(f"Looking for missing causal relationships (sample size: {sample_size})...")
        
        missing_relationships = []
        
        # Get problems that have symptoms but unclear causes
        problems_needing_causes = [
            key for key, data in self.problems.items() 
            if data['symptoms'] and not data['root_causes']
        ]
        
        # Sample problems to analyze
        import random
        sample_problems = random.sample(
            problems_needing_causes, 
            min(sample_size, len(problems_needing_causes))
        )
        
        for problem_key in sample_problems:
            problem_data = self.problems[problem_key]
            
            # Strategy 1: Use semantic similarity for quick relevant problem discovery
            relevant_candidates = self._find_semantically_similar_problems(problem_key, top_k=10)
            
            # Strategy 2: Also include random sample for non-obvious causal relationships
            all_other_problems = [p for p in self.problems.keys() if p != problem_key]
            random_candidates = random.sample(all_other_problems, min(5, len(all_other_problems)))
            
            # Combine candidates (remove duplicates)
            all_candidates = list(set(relevant_candidates + random_candidates))
            
            print(f"Analyzing {problem_key} against {len(all_candidates)} potential causes...")
            
            for potential_cause in all_candidates:
                # Check if relationship already exists
                if not self.causal_graph.has_edge(potential_cause, problem_key):
                    print(f"  Testing: {potential_cause} -> {problem_key}")
                    validation = self.validate_causal_relationship(potential_cause, problem_key)
                    
                    print(f"    Result: {validation['plausible']} (confidence: {validation['confidence']:.2f})")
                    
                    if validation['plausible'] and validation['confidence'] > 0.6:
                        print(f"    ✓ FOUND: {validation['reasoning'][:80]}...")
                        missing_relationships.append({
                            'cause': potential_cause,
                            'effect': problem_key,
                            'validation': validation,
                            'suggested': True,
                            'discovery_method': 'semantic' if potential_cause in relevant_candidates else 'random'
                        })
                    else:
                        print(f"    ✗ Not causal")
                    print()
        
        return missing_relationships
    
    def _find_semantically_similar_problems(self, problem_key: str, top_k: int = 10) -> List[str]:
        """Find semantically similar problems using simple text similarity."""
        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np
            
            # Initialize model
            model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Get text for target problem
            target_problem = self.problems[problem_key]
            target_text = f"{target_problem['title']} {target_problem['description']} {target_problem['symptoms']}"
            target_embedding = model.encode([target_text])[0]
            
            # Get embeddings for all other problems
            similarities = []
            for other_key, other_data in self.problems.items():
                if other_key != problem_key:
                    other_text = f"{other_data['title']} {other_data['description']} {other_data['root_causes']}"
                    other_embedding = model.encode([other_text])[0]
                    
                    # Cosine similarity
                    similarity = np.dot(target_embedding, other_embedding) / (
                        np.linalg.norm(target_embedding) * np.linalg.norm(other_embedding)
                    )
                    similarities.append((other_key, similarity))
            
            # Return top similar problems
            similarities.sort(key=lambda x: x[1], reverse=True)
            return [key for key, _ in similarities[:top_k]]
            
        except ImportError:
            print("⚠️  sentence-transformers not available, using random selection")
            import random
            all_other = [p for p in self.problems.keys() if p != problem_key]
            return random.sample(all_other, min(top_k, len(all_other)))
    
    def generate_report(self, validation_results: List[Dict], missing_relationships: List[Dict]) -> str:
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
        section_name = "Indicators" if section_type == 'symptoms' else "Causes"
        
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
        """Update markdown links to use hover tooltips for confidence/strength."""
        # Pattern to match any markdown link: [title](problem.md) with any optional trailing content
        link_pattern = r'\[([^\]]+)\]\(([^)]+)\.md\)(?:[^\n\r\-]*)?'
        
        def replace_link(match):
            title = match.group(1)
            problem_slug = match.group(2)
            
            # Find confidence and strength for this relationship
            for rel in relationships:
                if rel['cause'] == problem_slug:
                    confidence = rel.get('confidence', 0.0)
                    strength = rel.get('strength', confidence)  # fallback to confidence if strength not available
                    
                    # Format: [Title](link.md) <span class="info-tooltip" title="Confidence: 0.85, Strength: 0.92">ⓘ</span>
                    tooltip_text = f"Confidence: {confidence:.3f}, Strength: {strength:.3f}"
                    return f'[{title}]({problem_slug}.md) <span class="info-tooltip" title="{tooltip_text}">ⓘ</span>'
            
            # If no relationship data found, return just the clean link
            return f'[{title}]({problem_slug}.md)'
        
        return re.sub(link_pattern, replace_link, section_content)


def main():
    parser = argparse.ArgumentParser(
        description="Validate causal relationships using SD-SCM inspired approach",
        epilog="""Examples:
  %(prog)s --limit 10                      # Validate first 10 relationships
  %(prog)s --random-start --limit 5        # Validate 5 random relationships  
  %(prog)s --file feature-factory.md       # Validate relationships for specific problem
  %(prog)s --file bikeshedding.md --limit 3 # Validate 3 relationships for specific problem
  %(prog)s --update-files                  # Update problem files with results
  %(prog)s --random-start --update-files   # Random validation + file updates""",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output", default="causal_validation_report.md", help="Output report file (default: %(default)s)")
    parser.add_argument("--sample-missing", type=int, default=5, help="Sample size for missing relationship detection (default: %(default)s)")
    parser.add_argument("--limit", type=int, default=5, help="Limit number of relationships to validate (default: %(default)s)")
    parser.add_argument("--update-files", action="store_true", help="Update problem files with confidence scores and new relationships")
    parser.add_argument("--random-start", action="store_true", help="🎲 Start validation with random relationships instead of deterministic order (useful for varied sampling)")
    parser.add_argument("--file", type=str, help="Validate relationships for a specific problem file (e.g., 'feature-factory.md' or 'feature-factory')")
    args = parser.parse_args()
    
    validator = CausalRelationshipValidator()
    validator.load_problems()
    validator.build_causal_graph()
    
    # Validate existing relationships
    validation_results = validator.validate_all_relationships(limit=args.limit, random_start=args.random_start, specific_file=args.file)
    
    # Find missing causal relationships
    missing_relationships = validator.find_missing_causal_relationships(sample_size=args.sample_missing)
    
    # Generate report
    report = validator.generate_report(validation_results, missing_relationships)
    
    # Save report
    with open(args.output, 'w') as f:
        f.write(report)
    
    print(f"Validation report saved to {args.output}")
    
    # Update files with confidence scores if requested
    if args.update_files:
        validator.update_problem_files_with_confidence(validation_results, missing_relationships)


if __name__ == "__main__":
    main()