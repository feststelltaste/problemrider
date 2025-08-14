Review all existing problem descriptions and identify entries that describe essentially the same issue, even if expressed differently. When two or more problems appear to be closely related, propose merging them into a single, coherent problem entry. For each suggestion, preserve the most complete title, symptoms, and root cause—and if each version provides unique insight, weave those together meaningfully. Make sure any links pointing to the older versions are updated to reference the new, unified problem.

Before proceeding with merging, generate and share a list of candidate problems along with a brief rationale for each suggested group or pair. Only proceed with the actual merge once I have reviewed and approved it.

After completing the deduplication, run the automated analysis to update all causal relationships:
```bash
python scripts/validate_causal_relationships.py
```

This will automatically validate and update all causal links after the merge, ensuring the problem catalog remains consistent and well-connected.