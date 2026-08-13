Continue by creating at least 20 new problem descriptions for legacy software systems
using the existing pattern template. First, review `scripts/backlog/new.md` when it
exists and generate entries based on its contents. Then browse the current problem
descriptions and prioritize detailing issues mentioned in the **Symptoms** or
**Causes** sections when they warrant a separate reusable problem pattern.

Do not duplicate existing problem descriptions. Focus on development-related issues,
although organizational, business, or process-level problems are appropriate when
they emerge from development challenges. Avoid framing descriptions as “lack of” or
“no use of,” because those describe missing solutions rather than problematic states.
The solution space is addressed separately.

Treat every proposed Symptoms or Causes relationship as a causal hypothesis. Before
adding it, check that the direction is correct, state a plausible mechanism, consider
shared causes or mere correlation, qualify the context in which it applies, and look
for evidence independent of this catalog. AI-generated prose, semantic similarity,
and repetition in two files are not evidence.

For accepted relationships, represent the same directed claim consistently at both
ends: if A lists B under **Symptoms**, B should list A under **Causes**. This
reciprocity is required for graph consistency but does not confirm causal validity.
If a relationship is merely plausible and cannot be supported, report it as a
candidate instead of silently adding it. Use `/pr:review_causal_links` when a full
evidence-based assessment is needed.
