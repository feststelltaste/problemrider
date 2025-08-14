Continue by creating at least 20 new problem descriptions for legacy software systems using the existing pattern template. First, review the `scripts/backlog/new.md` file and generate entries based on its contents. Then, browse the current problem descriptions and prioritize detailing issues mentioned in the **Symptoms ▲** or **Causes ▼** sections—if they warrant elaboration. Make sure to link both ways between new and original entries. Do not duplicate problem descriptions that already exist. Focus on development-related issues, although it's fine to include organizational, business, or process-level problems if they emerge from development challenges. Avoid framing descriptions as "lack of" or "no use of," since those describe missing solutions rather than actual problems. The solution space will be addressed separately.

After creating new problems, run both scripts to establish relationships and generate symptoms:

1. **Find related problems** based on semantic similarity:
```bash
python scripts/calculate_related_problems.py --file new-problem-name
```

2. **Generate symptoms** from causal relationships (if any exist):
```bash
python scripts/validate_causal_relationships.py --test-run
```

The first script efficiently calculates semantic relationships for just the new problem. The second script will validate any causal relationships and generate symptoms if the new problem has validated root causes.

