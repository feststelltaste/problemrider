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
- **Deutsche Slugs:** Die Dateinamen (und damit URLs) in `_problems_de`/`_solutions_de` sind **eigene, aus dem deutschen Titel abgeleitete Slugs** (lowercase, Bindestriche, wie in `CLAUDE.md` für alle Markdown-Dateinamen gefordert) — sie sind **nicht** identisch mit dem englischen Dateinamen. Bestätigt vom Nutzer. Das ändert Entscheidung 1 (Dateibaum) und Entscheidung 6 (Sprachumschalter) unten gegenüber der ursprünglichen "gleicher Slug"-Annahme.

## Offene Architekturentscheidungen

### 1. Wie werden EN/DE-Inhalte strukturiert?

**Empfehlung:** Kein Übersetzungs-Plugin (`jekyll-polyglot` o.ä.), sondern zwei zusätzliche eigene Collections `problems_de` und `solutions_de`, die via `permalink` unter `/de/` ausgegeben werden, aber **eigene, aus dem deutschen Titel abgeleitete Dateinamen** tragen (siehe "Entschieden" oben):

```
_problems/foo.md                    (Englisch, bestehend, unverändert)
_problems_de/deutscher-slug.md      (Deutsch, neu, eigener Dateiname)
_solutions/bar.md                   (Englisch, bestehend, unverändert)
_solutions_de/anderer-slug.md       (Deutsch, neu, eigener Dateiname)
```

- `_config.yml`: neue Collection-Einträge mit `permalink: /de/problems/:name.html` bzw. `/de/solutions/:name.html`.
- `lang: de` in den Defaults für diese Collections, `lang: en` explizit (oder `site.lang`-Default) für die bestehenden.
- **Da der Dateiname/Slug zwischen EN und DE nicht mehr übereinstimmt, braucht jede DE-Datei ein Front-Matter-Feld, das auf ihr englisches Original zurückverweist**, z. B. `en_slug: foo`. Das Gegenstück lässt sich dann genau wie die bestehenden `related_problems`/`solutions`-Lookups per Liquid auflösen: `site.problems_de | where: "en_slug", page.slug | first` (EN→DE) bzw. `site.problems | where: "slug", page.en_slug | first` (DE→EN) — passt zum bereits in `problem.html`/`solution.html` verwendeten Muster, ohne die 1013 bestehenden englischen Dateien anfassen zu müssen.
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

`related_problems`, `related_solutions`, `solutions`/`problems`-Listen sowie die Landscape-Cluster basieren auf Embeddings des **englischen** Texts. Statt für Deutsch neu zu berechnen (zweiter Embedding-Lauf, zweite Landscape-Generierung, doppelte Pflege): Die deutschen Dateien übernehmen exakt dieselben (englischen) Slug-Listen aus ihrem englischen Original 1:1 — diese Felder bleiben also **englische Slugs**, auch in der DE-Datei. Weil DE-Dateinamen jetzt eigene Slugs haben (s. o.), müssen `_layouts/problem.html`/`solution.html` beim Auflösen dieser Listen sprachabhängig den passenden Join-Key wählen: auf EN-Seiten wie bisher `site.problems | where: "slug", related_slug`, auf DE-Seiten `site.problems_de | where: "en_slug", related_slug` (statt `"slug"`). Das ist ein kleiner, aber notwendiger Umbau der bestehenden Layout-Logik, siehe Phase 0/1. Landscape-Seite bleibt vorerst englisch-only; Sprachumschalter zeigt "Landscape" auf der deutschen Seite optional ausgegraut/verlinkt zur EN-Version, bis eine deutsche Landscape-Variante separat entschieden wird.

### 6. Wo lebt der Sprachumschalter?

Ein Link/Button im `header.html`, der zur Gegenstück-URL wechselt. Da EN- und DE-Slug nicht mehr identisch sind, kann die Zielseite nicht mehr aus dem aktuellen `page.slug` + geändertem Präfix berechnet werden, sondern muss über das `en_slug`-Feld nachgeschlagen werden (s. Entscheidung 1): auf einer EN-Problemseite `site.problems_de | where: "en_slug", page.slug | first`, auf einer DE-Problemseite direkt `site.problems | where: "slug", page.en_slug | first` (analog für Lösungen). Fehlt das Gegenstück (weil eine Datei noch nicht übersetzt ist), Umschalter ausblenden statt auf 404 zu verlinken. Für die vier Übersichtsseiten (`/`, `/problems/`, `/solutions/`, `/categories/`) reichen feste Paare (`/` ↔ `/de/`, etc.), da die überhaupt keinen Slug haben.

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
- [ ] Verbleibende Entscheidungen 2–5 oben bestätigen (1 und 6-Grundlage — URL-Präfix, deutsche Slugs — sind bereits entschieden)
- [ ] `_config.yml`: Collections `problems_de`/`solutions_de`, Defaults, `lang`-Feld
- [ ] Abschnitts-Erkennung auf Glyphen umstellen (Entscheidung 2), inkl. Regressionslauf von `check_links.py` und `validate_causal_links.py --detail` auf dem bestehenden englischen Bestand
- [ ] `_layouts/problem.html`/`solution.html`: sprachabhängige Auflösung der `related_problems`/`related_solutions`/`solutions`/`problems`-Listen umbauen (Join-Key `slug` vs. `en_slug`, s. Entscheidung 5)
- [ ] `_data/categories_de.yml` und `_data/i18n.yml` (oder `en.yml`/`de.yml`) anlegen
- [ ] Sprachumschalter in `header.html` bauen, Auflösung über `en_slug` (s. Entscheidung 6), zunächst auf Dummy-Zielen testbar

### Phase 1 — Chrome/UI übersetzen
- [ ] `_layouts/default.html`, `_layouts/problem.html`, `_layouts/solution.html`
- [ ] `_includes/header.html`, `footer.html`, `head.html`, `analysis-trail.html`
- [ ] `index.html`, `problems.html`, `solutions.html`, `categories.html`, `landscape.html` (deutsche Kopien unter `/de/`)
- [ ] UI-Strings in `analysis-trail.js`, `landscape.js` auf `_data/i18n.yml` umstellen
- [ ] `hreflang`-Alternates + Sitemap-Eintrag prüfen (`jekyll-seo-tag`, `jekyll-sitemap`)

Ergebnis: `/de/` ist erreichbar, navigierbar, aber Problem-/Lösungsseiten liefern noch 404 bzw. sind leer.

### Phase 2 — Titel-Übersetzungstabelle (Grundlage für Phase 3)
- [ ] Für alle 1013 Slugs den deutschen Titel + deutsche Kurzbeschreibung **und einen daraus abgeleiteten deutschen Slug** vorab erzeugen (z. B. per Skript/Agenten-Batch, Ergebnis in einer Zwischentabelle, etwa `scripts/i18n/titles_de.csv` oder `_data/titles_de.yml`, Spalten: `slug` (EN, Join-Key), `collection`, `title_de`, `description_de`, `slug_de` (neuer Dateiname, lowercase/Bindestriche))
- [ ] `slug_de`-Werte auf Eindeutigkeit prüfen (Kollisionen zweier unterschiedlicher EN-Slugs auf denselben deutschen Slug sind zu erwarten und müssen manuell aufgelöst werden, z. B. durch Anhängen eines unterscheidenden Worts)
- [ ] Titel manuell/durch Review gegenprüfen: Title-Case-Regeln gelten auch für die deutsche Variante? (zu klären — im Deutschen ist durchgehende Großschreibung der Substantive ungewöhnlich; siehe offene Frage unten)
- [ ] Diese Tabelle ist die einzige Quelle für Linktexte **und Ziel-Dateinamen** in Phase 3 — verhindert, dass zwei Dateien denselben Zielslug unterschiedlich übersetzen oder unterschiedliche Dateinamen für dasselbe Original vergeben

### Phase 3 — Inhalte übersetzen (der große Teil)
Batch-weise, z. B. 25–50 Dateien pro Durchgang, mit Fortschrittstabelle unten. Pro Datei:
- [ ] Neue Datei unter `_problems_de/<slug_de>.md` bzw. `_solutions_de/<slug_de>.md` anlegen (Dateiname = `slug_de` aus der Titeltabelle, **nicht** der englische Dateiname)
- [ ] Front Matter: `title`/`description` aus der Titeltabelle übernehmen, `lang: de` setzen, `en_slug: <englischer Slug>` setzen (Rückverweis, s. Entscheidung 1/6), `category`/`related_problems`/`related_solutions`/`solutions`/`problems` unverändert als englische Slugs übernehmen (Entscheidung 5)
- [ ] Fließtext übersetzen, Abschnittsstruktur/Glyphen 1:1 beibehalten
- [ ] Jeden internen Link umschreiben: Linktext = deutscher Titel aus der Tabelle, Pfad zeigt auf `<slug_de>.md` in der `_de`-Collection (aus der Titeltabelle nachschlagen, nicht den EN-Dateinamen wiederverwenden)
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
