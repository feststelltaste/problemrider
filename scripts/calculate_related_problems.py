#!/usr/bin/env python3
"""
Simple embeddings using sentence-transformers with Qwen3-Embedding-0.6B or local embedding service.
Install dependencies: pip install -r scripts/requirements.txt

Usage:
    python scripts/calculate_related_problems.py --dry-run
    python scripts/calculate_related_problems.py --use-local --local-url http://192.168.1.100:8000
    python scripts/calculate_related_problems.py --file new-problem  # Update specific problem only
    
Environment variables:
    LOCAL_EMBEDDING_URL: Default URL for local embedding service (default: http://host.docker.internal:1234)
    
Cache:
    Embeddings are cached in _embeddings/[problem-name].yaml files for faster subsequent runs.
"""

import yaml
import argparse
import re
import numpy as np
import os
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import markdown
    from bs4 import BeautifulSoup
except ImportError:
    print("❌ markdown/beautifulsoup4 not installed")
    print("Run: pip install -r scripts/requirements.txt")
    exit(1)


try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("❌ sentence-transformers not installed")
    print("Run: pip install sentence-transformers")
    exit(1)

try:
    import requests
except ImportError:
    requests = None
    print("⚠️  requests not installed - local embedding service unavailable")
    print("Run: pip install requests")



# Configuration
MIN_SIMILARITY = 0.5
# For WSL2 users: Use Windows host IP to access LM Studio running on Windows
DEFAULT_LOCAL_EMBEDDING_URL = "http://host.docker.internal:1234"


class SimpleEmbeddingAnalyzer:
    def __init__(self, problems_dir: str = "_problems", use_local: bool = False, local_url: str = None):
        self.problems_dir = Path(problems_dir)
        self.problems: Dict[str, Dict] = {}
        self.embeddings: Dict[str, np.ndarray] = {}
        self.use_local = use_local
        self.local_url = local_url or os.getenv('LOCAL_EMBEDDING_URL', DEFAULT_LOCAL_EMBEDDING_URL)
        self.local_model_name = None
        
        
        if self.use_local:
            if requests is None:
                raise ValueError("requests package is required for local embedding service")
            print(f"🌐 Using local embedding service at: {self.local_url}")
            self._test_local_service()
            self.model = None
        else:
            print("🤖 Loading Qwen3-Embedding-0.6B model locally...")
            self.model = SentenceTransformer('Qwen/Qwen3-Embedding-0.6B')
            
            device = self.model.device
            print(f"Using device: {device}")
            if 'cuda' in str(device):
                import torch
                gpu_name = torch.cuda.get_device_name(device)
                print(f"GPU: {gpu_name}")
            else:
                import platform
                print(f"CPU: {platform.processor()}")
    
    def _get_content_hash(self, problem_data: Dict) -> str:
        """Generate MD5 hash of problem content for cache invalidation."""
        import hashlib
        
        # Combine the content that affects embeddings
        content_parts = [
            problem_data['title'],
            problem_data['description'],
            problem_data['content_sections']['description'],
            problem_data['content_sections']['indicators']
        ]
        content_text = " ".join(part for part in content_parts if part.strip())
        
        return hashlib.md5(content_text.encode()).hexdigest()
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def _get_model_name(self) -> str:
        """Get the name of the embedding model being used."""
        if self.use_local:
            # Normalize local model names to match the standard naming
            if self.local_model_name:
                return self._normalize_model_name(self.local_model_name)
            return "unknown"
        else:
            return "Qwen/Qwen3-Embedding-0.6B"
    
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
    
    def _save_embedding_to_file(self, problem_key: str, embedding: np.ndarray, content_hash: str) -> None:
        """Save embedding and metadata to a separate YAML file in _embeddings directory."""
        # Ensure _embeddings directory exists
        embeddings_dir = Path("_embeddings")
        embeddings_dir.mkdir(exist_ok=True)
        
        # Create embedding file path
        embedding_file = embeddings_dir / f"{problem_key}.yaml"
        
        try:
            # Prepare embedding data
            embedding_data = {
                'hash': content_hash,
                'model': self._get_model_name(),
                'timestamp': self._get_timestamp(),
                'embedding': embedding.tolist()  # Put embedding as last entry
            }
            
            # Write YAML file with embedding values on separate lines (not as list)
            with open(embedding_file, 'w', encoding='utf-8') as f:
                # Write metadata first
                f.write(f"hash: {embedding_data['hash']}\n")
                f.write(f"model: {embedding_data['model']}\n")
                f.write(f"timestamp: {embedding_data['timestamp']}\n")
                f.write("embedding:\n")
                
                # Write embedding values under each other, not as a list
                for value in embedding_data['embedding']:
                    f.write(f"- {value}\n")
            
            print(f"💾 Saved embedding cache to _embeddings/{problem_key}.yaml")
                    
        except Exception as e:
            print(f"⚠️  Could not save embedding to _embeddings/{problem_key}.yaml: {e}")
    
    def _test_local_service(self):
        """Test connectivity to local embedding service and detect available models."""
        try:
            # Test LM Studio's models endpoint instead of /health
            response = requests.get(f"{self.local_url}/v1/models", timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data.get('data'):
                    model_names = [model.get('id', 'unknown') for model in data['data']]
                    print(f"✅ LM Studio is available with {len(model_names)} models: {', '.join(model_names[:3])}{'...' if len(model_names) > 3 else ''}")
                    
                    # Try to find the closest match to our preferred model
                    preferred_model = "text-embedding-qwen3-embedding-0.6b"
                    if preferred_model in model_names:
                        self.local_model_name = preferred_model
                        print(f"🎯 Using preferred model: {self.local_model_name}")
                    else:
                        # Look for any qwen3 0.6b variant
                        qwen_models = [m for m in model_names if 'qwen3' in m.lower() and '0.6b' in m.lower()]
                        if qwen_models:
                            self.local_model_name = qwen_models[0]
                            print(f"🎯 Using Qwen3 0.6B variant: {self.local_model_name}")
                        else:
                            # Fall back to first model
                            self.local_model_name = model_names[0]
                            print(f"⚠️  Preferred model not found, using first available: {self.local_model_name}")
                            print(f"💡 Available models: {', '.join(model_names)}")
                else:
                    print("⚠️  LM Studio is available but no models are loaded")
                    raise ValueError("No models available in local service")
            else:
                print(f"⚠️  Local service responded with status {response.status_code}")
                raise ValueError(f"Local service returned status {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"❌ Cannot connect to LM Studio: {e}")
            print("🔄 Falling back to local model...")
            self.use_local = False
            print("🤖 Loading Qwen3-Embedding-0.6B model locally...")
            self.model = SentenceTransformer('Qwen/Qwen3-Embedding-0.6B')
        
    def load_problems(self) -> None:
        """Load all problem files."""
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
                            'current_related_problems': metadata.get('related_problems', []),
                            'metadata': metadata,
                            'content_sections': self._extract_additional_sections(markdown_content)
                        }
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                
        print(f"Loaded {len(self.problems)} problems")
    
    def _clean_markdown_text(self, text: str) -> str:
        """Clean markdown formatting from text."""
        # Convert markdown to HTML, then extract plain text
        html = markdown.markdown(text)
        soup = BeautifulSoup(html, 'html.parser')
        plain_text = soup.get_text()
        return re.sub(r'\s+', ' ', plain_text).strip()
    
    def _extract_section(self, content: str, section_name: str) -> str:
        """Extract a specific section from markdown content."""
        pattern = rf'## {section_name}.*?\n(.*?)(?=\n## |\n$)'
        match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
        if match:
            return self._clean_markdown_text(match.group(1))
        return ''
    
    def _extract_additional_sections(self, content: str) -> dict:
        """Extract specific sections: indicators, description, and examples."""
        sections = {
            'indicators': self._extract_section(content, 'Indicators'),
            'description': self._extract_section(content, 'Description')
        }
        return sections
    
    def _get_embeddings_local(self, texts: List[str]) -> np.ndarray:
        """Get embeddings from LM Studio using OpenAI-compatible API with batch processing."""
        try:
            from tqdm import tqdm
            
            # Try batch processing first
            batch_size = min(20, len(texts))  # Start with reasonable batch size
            
            if len(texts) > 1:
                try:
                    # Test if batch processing is supported with 2 texts
                    test_batch = texts[:min(2, len(texts))]
                    test_response = requests.post(
                        f"{self.local_url}/v1/embeddings",
                        json={
                            "input": test_batch,
                            "model": self.local_model_name
                        },
                        headers={"Content-Type": "application/json"},
                        timeout=30
                    )
                    
                    if test_response.status_code == 200:
                        test_data = test_response.json()
                        if isinstance(test_data.get("data"), list) and len(test_data["data"]) == len(test_batch):
                            print(f"🚀 Using batch processing (batch_size={batch_size})")
                            return self._get_embeddings_batch(texts, batch_size)
                except Exception as e:
                    print(f"⚠️ Batch test failed: {e}")
                    pass  # Fall back to individual requests
            
            # Fall back to individual requests
            print("🐌 Using individual requests (batch processing not supported)")
            embeddings = []
            
            with tqdm(texts, desc="🌐 Getting embeddings", unit="text") as pbar:
                for text in pbar:
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
                    embeddings.append(data["data"][0]["embedding"])
            
            return np.array(embeddings)
        except requests.exceptions.RequestException as e:
            print(f"❌ Error getting embeddings from LM Studio: {e}")
            raise
    
    def _get_embeddings_batch(self, texts: List[str], batch_size: int) -> np.ndarray:
        """Get embeddings in batches for better performance."""
        from tqdm import tqdm
        embeddings = []
        
        # Process in batches
        num_batches = (len(texts) + batch_size - 1) // batch_size
        with tqdm(range(0, len(texts), batch_size), desc="🚀 Batch processing", unit="batch", total=num_batches) as pbar:
            for i in pbar:
                batch = texts[i:i + batch_size]
                pbar.set_postfix({"batch_size": len(batch)})
                
                response = requests.post(
                    f"{self.local_url}/v1/embeddings",
                    json={
                        "input": batch,
                        "model": self.local_model_name
                    },
                    headers={"Content-Type": "application/json"},
                    timeout=60  # Longer timeout for batches
                )
                response.raise_for_status()
                data = response.json()
                
                # Extract embeddings from batch response
                batch_embeddings = [item["embedding"] for item in data["data"]]
                embeddings.extend(batch_embeddings)
        
        return np.array(embeddings)
    
    def create_embeddings(self) -> None:
        """Create embeddings using cache when possible, only computing new/changed problems."""
        print("Creating embeddings (using cache when possible)...")
        
        texts_to_encode = []
        keys_to_encode = []
        cached_count = 0
        
        # Check each problem against cache in _embeddings directory
        for problem_key, problem_data in self.problems.items():
            content_hash = self._get_content_hash(problem_data)
            
            # Check if we have cached embedding in _embeddings directory
            embedding_file = Path("_embeddings") / f"{problem_key}.yaml"
            cached_embedding = None
            cached_hash = None
            cached_model = None
            current_model = self._get_model_name()
            
            if embedding_file.exists():
                try:
                    with open(embedding_file, 'r', encoding='utf-8') as f:
                        cache_data = yaml.safe_load(f)
                    
                    cached_embedding = cache_data.get('embedding')
                    cached_hash = cache_data.get('hash')
                    cached_model = cache_data.get('model')
                except Exception as e:
                    print(f"⚠️  Could not load cache from {embedding_file}: {e}")
            
            # Normalize model names for comparison
            normalized_cached = self._normalize_model_name(cached_model) if cached_model else None
            normalized_current = self._normalize_model_name(current_model)
            
            if (cached_embedding and 
                cached_hash == content_hash and 
                normalized_cached == normalized_current):
                # Use cached embedding from separate file
                if isinstance(cached_embedding, list):
                    cached_embedding = np.array(cached_embedding)
                self.embeddings[problem_key] = cached_embedding
                cached_count += 1
            else:
                # Need to compute new embedding
                text_parts = [
                    problem_data['title'],
                    problem_data['description'],  # YAML description
                    problem_data['content_sections']['description'],  # Description section
                    problem_data['content_sections']['indicators']
                ]
                # Filter out empty parts
                text_parts = [part for part in text_parts if part.strip()]
                text = " ".join(text_parts)
                texts_to_encode.append(text)
                keys_to_encode.append(problem_key)
        
        print(f"📁 Using {cached_count} cached embeddings")
        
        if texts_to_encode:
            print(f"🆕 Computing {len(texts_to_encode)} new embeddings...")
            
            if self.use_local:
                print(f"🌐 Generating embeddings via local service at {self.local_url}")
                new_embeddings = self._get_embeddings_local(texts_to_encode)
            else:
                print("🤖 Generating embeddings using local Qwen3-Embedding-0.6B model")
                new_embeddings = self.model.encode(texts_to_encode, show_progress_bar=True, convert_to_numpy=True)
            
            # Store new embeddings and update files
            for key, embedding in zip(keys_to_encode, new_embeddings):
                self.embeddings[key] = embedding
                
                # Update the file with embedded cache
                content_hash = self._get_content_hash(self.problems[key])
                self._save_embedding_to_file(key, embedding, content_hash)
        else:
            print("✅ All embeddings loaded from cache")
        
        print(f"Ready with embeddings for {len(self.embeddings)} problems")
    
    def find_related(self, problem_key: str, top_k: int = 6, min_similarity: float = 0.3) -> List[Tuple[str, float]]:
        """Find related problems using cosine similarity."""
        if problem_key not in self.embeddings:
            return []
        
        base_emb = self.embeddings[problem_key]
        similarities = []
        
        for other_key, other_emb in self.embeddings.items():
            if other_key != problem_key:
                # Cosine similarity
                similarity = np.dot(base_emb, other_emb) / (np.linalg.norm(base_emb) * np.linalg.norm(other_emb))
                if similarity >= min_similarity:
                    similarities.append((other_key, float(similarity)))
        
        return sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]
    
    def update_all_files(self, dry_run: bool = True, min_similarity: float = 0.3, specific_file: str = None):
        """Update all files with new related problems."""
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
                return
            
            problem_keys = [file_slug]
            print(f"{'[DRY RUN] ' if dry_run else ''}Updating specific file: {specific_file}")
        else:
            problem_keys = list(self.problems.keys())
            print(f"{'[DRY RUN] ' if dry_run else ''}Updating all files...")
        
        updates = 0
        for problem_key in problem_keys:
            related = self.find_related(problem_key, top_k=6, min_similarity=min_similarity)
            # Create list of objects with slug and similarity (as quantized decimals)
            new_related = [{"slug": key, "similarity": round(score * 20) / 20} for key, score in related]
            
            if new_related:
                updates += 1
                # Show all updates with actual related problems
                related_with_scores = [f"{key} ({round(score * 20) / 20:.2f})" for key, score in related]
                print(f"{problem_key}:")
                print(f"  New: {related_with_scores}")
                print()
                
                if not dry_run:
                    self._update_file(problem_key, new_related)
        
        if specific_file:
            print(f"Updated 1 specific file" if updates > 0 else "No updates needed for specific file")
        else:
            print(f"Total: {updates} files would be updated")
    
    def _update_file(self, problem_key: str, new_related: List[dict]):
        """Update a single file."""
        problem_data = self.problems[problem_key]
        file_path = problem_data['file_path']
        
        with open(file_path, 'r') as f:
            content = f.read()
        
        parts = content.split('---', 2)
        if len(parts) >= 3:
            metadata = yaml.safe_load(parts[1])
            metadata['related_problems'] = new_related
            new_yaml = yaml.dump(metadata, default_flow_style=False, sort_keys=False)
            new_content = f"---\n{new_yaml}---{parts[2]}"
            
            with open(file_path, 'w') as f:
                f.write(new_content)

def main():
    parser = argparse.ArgumentParser(description="Generate related problems using Qwen3-Embedding-0.6B")
    parser.add_argument("--dry-run", action="store_true", help="Show changes without writing")
    parser.add_argument("--use-local", action="store_true", help="Use local embedding service instead of sentence-transformers")
    parser.add_argument("--local-url", type=str, help=f"URL of local embedding service (default: {DEFAULT_LOCAL_EMBEDDING_URL})")
    parser.add_argument("--file", type=str, help="Process only specific problem file (e.g., 'new-problem' or 'new-problem.md')")
    args = parser.parse_args()
    
    analyzer = SimpleEmbeddingAnalyzer(
        use_local=args.use_local,
        local_url=args.local_url
    )
    analyzer.load_problems()
    analyzer.create_embeddings()

    analyzer.update_all_files(dry_run=args.dry_run, min_similarity=MIN_SIMILARITY, specific_file=args.file)

if __name__ == "__main__":
    main()