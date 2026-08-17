# Plan: Deutsche Übersetzung von ProblemRider

Status: **Planungsphase — 0 / 1013 Katalog-Dateien übersetzt**

Dieses Dokument hält fest, wie ProblemRider (Layouts, Seiten, `_problems/`, `_solutions/`) ins Deutsche übersetzt werden kann, ohne die bestehende englische Seite, ihre Verlinkungslogik oder die Wartungs-Skripte zu brechen. Es ist als lebendes Arbeitsdokument gedacht (analog zu `plans/causal-link-review.md`): Entscheidungen zuerst treffen, dann in Batches abarbeiten und den Fortschritt hier abhaken.

## Ziel

Eine deutschsprachige Variante der Seite, die parallel zur englischen existiert (nicht als Ersatz), inklusive Navigation, Startseite, Kategorien- und Landscape-Seite sowie aller Problem-/Lösungs-Einträge.

## Nicht-Ziele (erstmal)

- Maintainer-/Repo-Dokumente (`CLAUDE.md`, `ARCH.md`, `README.md`, `MISTAKES.md`, `TECHDEBT.md`, `SOLUTIONS_PLAN.md`, `problem-pattern-template.md`, `solution-pattern-template.md`, `quality-tactics-reference.md`) — die stehen in `_config.yml` unter `exclude:` und sind nicht Teil der veröffentlichten Seite. Bleiben Englisch, es sei denn, das wird später explizit gewünscht.
- Automatische Spracherkennung/-Weiterleitung im Browser — MVP ist ein manueller Sprachumschalter.
- Ein neu berechnetes deutsches Embedding-/Ähnlichkeits-Modell für Landscape & "Related Problems/Solutions" (siehe Entscheidung 5 unten — wir spiegeln stattdessen die englischen Relationen).

## Kernproblem, das die Architektur bestimmt

Laut `CLAUDE.md` müssen interne Links den **exakten Titel** der Zielseite als Linktext verwenden. Übersetzen wir Titel ins Deutsche, muss also auch jeder Link, der auf diesen Titel verweist, in der deutschen Version umgeschrieben werden — in allen 1013 Dateien, die sich gegenseitig zitieren. Das bestimmt die gesamte Strategie unten: Übersetzung kann nicht datei-für-datei isoliert passieren, sondern braucht zuerst eine vollständige Titel-Übersetzungstabelle (Slug → deutscher Titel), bevor Fließtext/Links übersetzt werden.

Zweites Kernproblem: Mehrere Skripte und ein Client-seitiges Script matchen Abschnitte über den **exakten englischen Überschriftentext** (`## Symptoms ▲`, `## Causes ▼`, siehe `scripts/validate_causal_links.py:43-44` und den Inline-`<script>` in `_layouts/problem.html`, der nach `'Symptoms ▲'` / `'Root Causes ▼'` sucht). Übersetzte Überschriften wie `## Symptome ▲` würden diese Erkennung stillschweigend brechen. Muss vor der Massenübersetzung behoben werden (siehe Entscheidung 2).

## Entschieden

- **URL-Präfix `de/`:** Die deutsche Seite lebt unter `/de/...` (z. B. `/de/problems/foo.html`, `/de/`, `/de/solutions/`, `/de/categories/`), die englische bleibt unverändert auf der bestehenden Root-URL. Bestätigt vom Nutzer.

## Offene Architekturentscheidungen

### 1. Wie werden EN/DE-Inhalte strukturiert?

**Empfehlung:** Kein Übersetzungs-Plugin (`jekyll-polyglot` o.ä.), sondern zwei zusätzliche eigene Collections `problems_de` und `solutions_de`, die dieselben Dateinamen/Slugs wie ihre englischen Pendants verwenden und via `permalink` unter `/de/` ausgegeben werden:

```
_problems/foo.md        (Englisch, bestehend, unverändert)
_problems_de/foo.md     (Deutsch, neu)
_solutions/bar.md       (Englisch, bestehend, unverändert)
_solutions_de/bar.md    (Deutsch, neu)
```

- `_config.yml`: neue Collection-Einträge mit `permalink: /de/problems/:name.html` bzw. `/de/solutions/:name.html`.
- `lang: de` in den Defaults für diese Collections, `lang: en` explizit (oder `site.lang`-Default) für die bestehenden.
- Gleicher Slug = trivial berechenbares Gegenstück für Sprachumschalter und `hreflang`-Tags (`/problems/foo.html` ↔ `/de/problems/foo.html`).
- Warum kein Plugin: `jekyll-polyglot`/`jekyll-multiple-languages-plugin` gehen von 1:1 gespiegelten Dateibäumen mit eigenem URL-Rewriting aus und bringen ihre eigene Slug-/Permalink-Logik mit, die mit den bestehenden custom `permalink`-Patterns und `jekyll-relative-links` kollidieren kann. Zwei eigene Collections sind mehr Handarbeit in `_config.yml`, aber vollständig unter unserer Kontrolle und ändern nichts an der bestehenden `_problems`/`_solutions`-Pipeline (Sync-Skripte, Embeddings, Landscape).
- Der Build läuft über eine eigene GitHub-Actions-Pipeline (`.github/workflows/jekyll.yml`, `bundle exec jekyll build`), nicht über die eingeschränkte `github-pages`-Gem-Whitelist — ein Plugin wäre also technisch möglich, wird hier aber trotzdem nicht empfohlen (s.o.).

### 2. Abschnitts-Erkennung sprachunabhängig machen

Vor jeder Inhaltsübersetzung: `SYMPTOMS_HEADING`/`CAUSES_HEADING` in `scripts/validate_causal_links.py` (und die analoge Suche in `scripts/check_links.py`, `scripts/sync_problem_solution_links.py`, sowie das Inline-Script in `_layouts/problem.html`) so umbauen, dass sie **auf das Glyphen-Suffix matchen** (`▲`, `▼`, `⟡`, `◆`, `⇄`, `○`), nicht auf den englischen Wortlaut davor. Beispiel: Regex von `^## Symptoms ▲\n` auf `^## .*▲\s*\n` ändern. Die Glyphen bleiben in beiden Sprachen identisch (sie sind bereits Teil des Formats laut `problem-pattern-template.md`), das macht Übersetzung der Überschriften selbst risikofrei.

Nebenbefund während der Analyse: `problem.html` sucht nach `'Root Causes ▼'`, die Skripte nach `'Causes ▼'` — schon heute inkonsistent. Beim Umbau gleich vereinheitlichen.

### 3. Kategorien

Die 15 Kategorien (Process, Architecture, Code, …) bleiben intern als **englische Keys** in `category:` im Front Matter (damit Filter-URLs wie `/categories/#Architecture` und die JS-Filterlogik in `problems.html`/`solutions.html`/`categories.html` unverändert funktionieren). Für die Anzeige: eine `_data/categories_de.yml` mit Key→deutschem Label-Mapping, das die Layouts/Seiten per Include/Filter nachschlagen, wenn `page.lang == "de"`.

### 4. UI-Strings (Chrome)

Alle sichtbaren Strings in `_layouts/*.html`, `_includes/*.html`, `index.html`, `problems.html`, `solutions.html`, `categories.html`, `landscape.html` sowie hartcodierte UI-Strings in `assets/js/analysis-trail.js` und `assets/js/landscape.js` (Buttons, Tooltips, ARIA-Labels, "Show more"-Texte) brauchen deutsche Gegenstücke. Ansatz: kleine `_data/i18n.yml` (oder `_data/en.yml`/`_data/de.yml`) mit Key/Value-Paaren, dazu ein `t`-Include oder Liquid-Filter, das anhand von `page.lang` (Default `site.lang`) den richtigen String zieht. JS-seitige Strings brauchen ein kleines JS-Objekt mit denselben Keys, gespeist aus `<body data-lang="{{ page.lang }}">` o.ä.

### 5. Related Problems/Solutions & Landscape

`related_problems`, `related_solutions`, `solutions`/`problems`-Listen sowie die Landscape-Cluster basieren auf Embeddings des **englischen** Texts. Statt für Deutsch neu zu berechnen (zweiter Embedding-Lauf, zweite Landscape-Generierung, doppelte Pflege): Die deutschen Dateien übernehmen exakt dieselben Slug-Listen aus ihrem englischen Original (nur die angezeigten Titel/Beschreibungen kommen aus der jeweiligen deutschen Collection zur Laufzeit). Landscape-Seite bleibt vorerst englisch-only; Sprachumschalter zeigt "Landscape" auf der deutschen Seite optional ausgegraut/verlinkt zur EN-Version, bis eine deutsche Landscape-Variante separat entschieden wird.

### 6. Wo lebt der Sprachumschalter?

Ein Link/Button im `header.html`, der zur Gegenstück-URL wechselt (gleicher Slug, anderes Präfix). Für Collection-Seiten (`page.collection == 'problems'/'solutions'`) berechenbar aus `page.slug`; für die vier Übersichtsseiten (`/`, `/problems/`, `/solutions/`, `/categories/`) feste Paare (`/` ↔ `/de/`, etc.).

## Umfang (Ist-Stand)

| Bereich | Anzahl | Aufwand-Einschätzung |
|---|---|---|
| `_problems/*.md` | 452 Dateien | groß, dominiert das Projekt |
| `_solutions/*.md` | 561 Dateien | groß, dominiert das Projekt |
| Layouts (`_layouts/`) | 3 Dateien | klein |
| Includes (`_includes/`) | 4 Dateien | klein |
| Root-Seiten (`index.html`, `problems.html`, `solutions.html`, `categories.html`, `landscape.html`) | 5 Dateien | klein–mittel |
| JS-UI-Strings (`analysis-trail.js`, `landscape.js`) | 2 Dateien | mittel (Strings identifizieren, aus Code lösen) |
| `_config.yml`, `_data/` | neu anzulegen | klein |

Gesamt ca. **1013 Katalog-Dateien** plus ca. 14 Chrome-Dateien.

## Phasen

### Phase 0 — Vorbereitung (Voraussetzung für alles Weitere)
- [ ] Entscheidungen 1–6 oben bestätigen (bzw. mit Nutzer klären, falls Einwände)
- [ ] `_config.yml`: Collections `problems_de`/`solutions_de`, Defaults, `lang`-Feld
- [ ] Abschnitts-Erkennung auf Glyphen umstellen (Entscheidung 2), inkl. Regressionslauf von `check_links.py` und `validate_causal_links.py --detail` auf dem bestehenden englischen Bestand
- [ ] `_data/categories_de.yml` und `_data/i18n.yml` (oder `en.yml`/`de.yml`) anlegen
- [ ] Sprachumschalter in `header.html` bauen (zunächst auf Dummy-Zielen testbar)

### Phase 1 — Chrome/UI übersetzen
- [ ] `_layouts/default.html`, `_layouts/problem.html`, `_layouts/solution.html`
- [ ] `_includes/header.html`, `footer.html`, `head.html`, `analysis-trail.html`
- [ ] `index.html`, `problems.html`, `solutions.html`, `categories.html`, `landscape.html` (deutsche Kopien unter `/de/`)
- [ ] UI-Strings in `analysis-trail.js`, `landscape.js` auf `_data/i18n.yml` umstellen
- [ ] `hreflang`-Alternates + Sitemap-Eintrag prüfen (`jekyll-seo-tag`, `jekyll-sitemap`)

Ergebnis: `/de/` ist erreichbar, navigierbar, aber Problem-/Lösungsseiten liefern noch 404 bzw. sind leer.

### Phase 2 — Titel-Übersetzungstabelle (Grundlage für Phase 3)
- [ ] Für alle 1013 Slugs den deutschen Titel + deutsche Kurzbeschreibung vorab erzeugen (z. B. per Skript/Agenten-Batch, Ergebnis in einer Zwischentabelle, etwa `scripts/i18n/titles_de.csv` oder `_data/titles_de.yml`, Spalten: `slug`, `collection`, `title_de`, `description_de`)
- [ ] Titel manuell/durch Review gegenprüfen: Title-Case-Regeln gelten auch für die deutsche Variante? (zu klären — im Deutschen ist durchgehende Großschreibung der Substantive ungewöhnlich; siehe offene Frage unten)
- [ ] Diese Tabelle ist die einzige Quelle für Linktexte in Phase 3 — verhindert, dass zwei Dateien denselben Zielslug unterschiedlich übersetzen

### Phase 3 — Inhalte übersetzen (der große Teil)
Batch-weise, z. B. 25–50 Dateien pro Durchgang, mit Fortschrittstabelle unten. Pro Datei:
- [ ] Front Matter: `title`/`description` aus der Titeltabelle übernehmen, `lang: de` setzen, `category`/`related_problems`/`related_solutions`/`solutions`/`problems` unverändert (Slugs) übernehmen (Entscheidung 5)
- [ ] Fließtext übersetzen, Abschnittsstruktur/Glyphen 1:1 beibehalten
- [ ] Jeden internen Link umschreiben: Linktext = deutscher Titel aus der Tabelle, Pfad zeigt auf die `_de`-Collection
- [ ] Nach jedem Batch: `python scripts/check_links.py` und `python scripts/validate_causal_links.py --detail` auf die neue Collection anwenden (Skripte müssen dafür ein Verzeichnis-Argument bekommen statt hart `_problems`/`_solutions` anzunehmen — ggf. kleiner Parametrisierungs-Task vorab)

Empfehlung zur Ausführung: sobald Phase 0–2 stehen, eignet sich dieser Teil für eine pipeline-artige Bearbeitung (Datei → Übersetzen → Link-Validierung) über mehrere Agenten/Durchläufe, weil es sich um 1013 weitgehend gleichförmige, aber inhaltlich zu prüfende Einheiten handelt. Sollte der Nutzer das ausdrücklich wünschen, kann das über das Workflow-Tool orchestriert werden; das ist aber eine Ausführungsentscheidung für später, nicht Teil dieses Plans.

### Phase 4 — Qualitätssicherung
- [ ] Stichprobenartige fachliche Prüfung der Übersetzung (Terminologie konsistent? z. B. "Legacy System", "Technical Debt" — bewusst Anglizismus lassen oder eindeutschen? → offene Frage)
- [ ] Vollständigkeitscheck: jede `_problems/*.md` hat ein Pendant in `_problems_de/`, jede `_solutions/*.md` ein Pendant in `_solutions_de/`
- [ ] Build-Test: `bundle exec jekyll build` (voller Build, kein `--incremental`, da neue Collections/Content) + Stichproben-Navigation auf `/de/`

### Phase 5 — Laufender Betrieb
- [ ] Workflow-Regeln (`pr:generate_new_problems`, `pr:add_tech_debt`, manuelles Hinzufügen) so ergänzen, dass neue EN-Einträge eine offene ToDo-Markierung für die DE-Übersetzung bekommen (z. B. Eintrag in einer Rückstandsliste), statt dass `_problems_de`/`_solutions_de` stillschweigend veraltet
- [ ] Turnusmäßiger Nachzieh-Lauf (analog zu `calculate_related_problems.py`) für neu hinzugekommene Dateien

## Offene Fragen (vor Phase 2 zu klären)

1. **Title Case im Deutschen:** Die Titel-Case-Regel (NYT Manual of Style) ist eine englische Konvention. Deutsche Titel folgen üblicherweise normaler Satz-/Titelgroßschreibung (nur Substantive + Satzanfang groß). Vorschlag: für `_de`-Collections deutsche Standard-Großschreibung verwenden, nicht die NYT-Regel erzwingen — bitte bestätigen.
2. **Terminologie-Glossar:** Etablierte Fachbegriffe (Legacy System, Technical Debt, Root Cause, Code Smell, Refactoring, Onboarding …) — 1:1 im Original belassen oder eindeutschen? Empfehlung: gängige, im deutschen Software-Engineering-Sprachgebrauch etablierte Anglizismen beibehalten (z. B. "Legacy-System", "Refactoring", "Onboarding"), nur dort eindeutschen, wo ein ebenso gebräuchliches deutsches Wort existiert (z. B. "Root Cause" → "Grundursache"). Ein kurzes Glossar vor Phase 3 als `_data/glossary_de.yml` festhalten, damit alle Batches dieselben Begriffe konsistent verwenden.
3. **Umfang der Landscape-Seite auf Deutsch:** vorerst ausgeklammert (s. Entscheidung 5) — reicht das, oder soll sie von Anfang an mit übersetzten Labels mitgezogen werden?
4. **MVP vs. Vollausbau:** Reicht als erster Wurf eine deutsche Chrome-Übersetzung (Phase 0–1) plus eine kleine Teilmenge an Problemen/Lösungen (z. B. die am stärksten vernetzten 50) als Machbarkeitsnachweis, bevor alle 1013 Dateien übersetzt werden? Empfehlung: ja, um Format/Tooling/Terminologie an einer überschaubaren Menge zu validieren, bevor der große Batch läuft.

## Fortschritt Phase 3 (wird bei Bearbeitung aktualisiert)

| Batch | Umfang | Status |
|---|---|---|
| — | noch nicht gestartet | — |

*(Tabelle wird um Zeilen mit Slug-Bereichen/Dateilisten und Datum ergänzt, sobald Phase 3 beginnt.)*
