# Plan: Deutsche Übersetzung von ProblemRider

Status: **Phase 0 + Phase 1 + Phase 2 abgeschlossen (komplette Chrome/UI-Übersetzung, alle 5 Übersichtsseiten unter `/de/`, `_data/titles_de.yml` mit deutschem Titel/Beschreibung/Slug für alle 1013 Einträge) — Phase 3 läuft: 237 / 452 Problem-Dateien inhaltlich übersetzt (Batches 1–30), 0 / 561 Lösungs-Dateien. Autonome Batch-Bearbeitung (je ~8 Dateien, automatisch committet und gepusht) läuft weiter.**

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
- **Title Case:** Deutsche Titel folgen normaler deutscher Groß-/Kleinschreibung (nur Substantive, Satzanfänge, Eigennamen groß) — **nicht** der NYT-Title-Case-Regel aus `CLAUDE.md`, die eine rein englische Konvention ist. `convert_titles.py` gilt also nicht für `_de`-Titel; die deutsche Titeltabelle (Phase 2) braucht keine Title-Case-Prüfung. Bestätigt vom Nutzer.
- **Anglizismen bleiben:** Etablierte englische Fachbegriffe (Legacy System, Technical Debt, Root Cause, Code Smell, Refactoring, Onboarding, Feature Flag, …) werden im deutschen Fließtext **beibehalten**, nicht eingedeutscht — keine Fall-für-Fall-Entscheidung nötig. Bestätigt vom Nutzer. Ein schlankes `_data/glossary_de.yml` ist trotzdem sinnvoll, aber nur um Schreibweise/Grammatik konsistent zu halten (z. B. einheitlich "Legacy-System" mit Bindestrich, einheitliche Groß-/Kleinschreibung bei eingebetteten Fachbegriffen), nicht um zu entscheiden, ob übersetzt wird.
- **Vollständiger Umfang, kein MVP zuerst:** Es wird direkt der komplette Katalog übersetzt (alle 452 Probleme + 561 Lösungen), keine kleinere Teilmenge als Machbarkeitsnachweis vorab. Bestätigt vom Nutzer. Phase 3 bleibt trotzdem batchweise organisiert (siehe dort) — das ist nur zur Fortschrittskontrolle, keine Umfangs-Begrenzung.
- **Landscape-Seite wird mitübersetzt:** `/de/landscape/` bekommt eine deutsche Version. Die zugrundeliegende Cluster-Berechnung (UMAP/k-means auf den Embeddings, `scripts/create_landscape.py`) wird **nicht** neu für Deutsch gerechnet — Positionen/Cluster bleiben identisch zur englischen Landscape, nur die angezeigten Labels/Titel werden aus `_data/titles_de.yml`/den `_de`-Collections gezogen (Layout-technisch: `landscape.js` bzw. die Datengenerierung braucht einen deutschen Label-Layer über denselben Koordinaten). Bestätigt vom Nutzer, siehe Entscheidung 5 unten für Details.

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

`related_problems`, `related_solutions`, `solutions`/`problems`-Listen sowie die Landscape-Cluster basieren auf Embeddings des **englischen** Texts. Statt für Deutsch neu zu berechnen (zweiter Embedding-Lauf, zweite Landscape-Generierung, doppelte Pflege): Die deutschen Dateien übernehmen exakt dieselben (englischen) Slug-Listen aus ihrem englischen Original 1:1 — diese Felder bleiben also **englische Slugs**, auch in der DE-Datei. Weil DE-Dateinamen jetzt eigene Slugs haben (s. o.), müssen `_layouts/problem.html`/`solution.html` beim Auflösen dieser Listen sprachabhängig den passenden Join-Key wählen: auf EN-Seiten wie bisher `site.problems | where: "slug", related_slug`, auf DE-Seiten `site.problems_de | where: "en_slug", related_slug` (statt `"slug"`). Das ist ein kleiner, aber notwendiger Umbau der bestehenden Layout-Logik, siehe Phase 0/1.

**Landscape (Entscheidung: wird übersetzt, s. "Entschieden" oben):** `assets/js/landscape-data.js` wird nicht neu generiert — die x/y-Positionen und Cluster kommen unverändert aus dem englischen Lauf von `create_landscape.py`. Für `/de/landscape/` braucht `landscape.js` (oder die Datengenerierung) zusätzlich einen Titel-Lookup pro Knoten, der bei `lang: de` den deutschen Titel aus der Phase-2-Titeltabelle statt des englischen `title`-Felds zieht — die Koordinaten je Slug bleiben gleich, nur das angezeigte Label wechselt. Umsetzung dafür einplanen (kleiner Zusatzschritt in Phase 1 oder Phase 3, je nachdem wann die Titeltabelle steht).

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

### Phase 0 — Vorbereitung (Voraussetzung für alles Weitere) — ✅ abgeschlossen
- [x] Entscheidungen 1–6 umgesetzt, keine Einwände aufgekommen
- [x] `_config.yml`: Collections `problems_de`/`solutions_de` (Permalinks unter `/de/`), Defaults, `lang`-Feld (`en`/`de`), `site.lang: en`; leere `_problems_de`/`_solutions_de`-Verzeichnisse mit `.gitkeep` angelegt
- [x] Abschnitts-Erkennung auf Glyphen umgestellt (Entscheidung 2): `scripts/validate_causal_links.py` (`SYMPTOMS_GLYPH`/`CAUSES_GLYPH`) und das Show-more-Script in `_layouts/problem.html` matchen jetzt auf `▲`/`▼` statt auf den englischen Wortlaut; dabei auch den Bestandsbug behoben, dass die JS-Prüfung nach `'Root Causes ▼'` statt `'Causes ▼'` suchte und die Causes-Sektion nie traf. Regressionslauf von `validate_causal_links.py` vor/nach der Änderung ergab identische Zahlen (452 Probleme, 3447 Claims)
- [x] `_layouts/problem.html`/`solution.html`: sprachabhängige Auflösung der `related_problems`/`related_solutions`/`solutions`/`problems`-Listen umgebaut (Join-Key `slug` auf EN-Seiten, `en_slug` auf DE-Seiten, s. Entscheidung 5); der EN-Zweig ist Zeile für Zeile identisch zum bisherigen Code, also kein Verhaltensunterschied für die heutige englische Seite
- [x] `_data/categories_de.yml` (15 Kategorie-Labels, Keys gegen den tatsächlichen Bestand verifiziert) und `_data/i18n.yml` (109 EN/DE-Schlüssel für Chrome-Strings, Parität geprüft) angelegt — noch nicht in Layouts/Seiten verdrahtet, das ist Phase 1
- [x] Sprachumschalter in `header.html` gebaut: Collection-Seiten lösen über `en_slug` auf, Übersichtsseiten über Swap des `/de/`-Präfixes plus Existenzprüfung in `site.pages`; ohne Gegenstück bleibt der Link ausgeblendet (aktuell also überall, bis Phase 1/3 Inhalte liefern)

Nicht build-getestet (kein `bundle exec jekyll build` ausgeführt, siehe Projektregel zu Builds während der Iteration in `CLAUDE.md`) — Layout-Änderungen wurden von Hand geprüft; der englische Pfad ist unverändert, ein voller Build-Test ist ohnehin Teil von Phase 4.

### Phase 1 — Chrome/UI übersetzen — ✅ abgeschlossen
- [x] `_layouts/problem.html`, `solution.html`: Speed-nav, Related/Possible/Similar/Addressed-Überschriften, Back-Links (inkl. `/de/`-Zielpfade), Kategorie-Labels und die JS-Show-more/-less-Texte (jetzt über ein `%{count}`-Template statt hartcodiertem Satz) laufen über `site.data.i18n[current_lang]`
- [x] `_includes/header.html`: Nav-Labels + `/de/`-Präfix, Site-Title-Link zeigt auf die Startseite der aktuellen Sprache; `_includes/footer.html`: GitHub-Link/Lizenzzeile/Tagline übersetzt (neuer `footer.tagline`-Key); `_includes/analysis-trail.html`: jeder String inkl. aller aria-labels übersetzt, eingebetteter JSON-Katalog liest jetzt `site.problems_de`/`site.solutions_de` auf deutschen Seiten
- [x] `index.html`, `problems.html`, `solutions.html`, `categories.html`, `landscape.html`: deutsche Kopien unter `de/` angelegt (`/de/`, `/de/problems/`, `/de/solutions/`, `/de/categories/`, `/de/landscape/`); alle iterieren `site.problems_de`/`site.solutions_de` statt der englischen Collections. `de/landscape.html`s `<style>`-Block ist Byte-für-Byte identisch zum Original (verifiziert per `diff`) — nur das Markup wurde übersetzt. Bekannte Lücke: die Knoten-Labels auf der Landscape-Karte selbst (und die Legende-Chips, die Kategorienamen zeigen) kommen weiterhin aus `landscape-data.js`/`CATEGORY_COLORS`-Keys (englische Titel aus dem einmaligen Embedding-Lauf) — ein deutscher Label-Layer für die Karte ist eigene Folgearbeit, im Code kommentiert
- [x] UI-Strings in `analysis-trail.js`/`landscape.js` auf `_data/i18n.yml` umgestellt: ~25 dynamisch erzeugte Strings (Menü-Labels, aria-labels, Tooltips, Alerts) laufen jetzt über `window.ANALYSIS_TRAIL_I18N`/`window.LANDSCAPE_I18N` (mit englischem Literal-Fallback im JS selbst). Dabei denselben Symptoms/Causes-Erkennungsbug wie in Entscheidung 2 an zwei weiteren Stellen gefunden und behoben (`edgeForLink`, `localReferences` in `analysis-trail.js` matchten noch auf den englischen Wortlaut) — jetzt glyphenbasiert wie überall sonst
- [x] `hreflang`-Alternates ergänzt: `_includes/head.html` emittiert `<link rel="alternate" hreflang="...">` fürs Sprachenpaar plus `x-default`, per gleicher `en_slug`/`/de/`-Präfix-Logik wie der Sprachumschalter, leer wenn kein Gegenstück existiert. `jekyll-sitemap` brauchte keine Konfigänderung — erfasst neue Collections/Seiten automatisch über `output: true`

`_data/i18n.yml` ist im Zuge dessen auf 149 EN/DE-Schlüsselpaare gewachsen (Parität nach jeder Ergänzung geprüft). Alle neuen/geänderten Dateien wurden von Hand auf Liquid-Tag-Balance, Front-Matter-Validität (`ruby -ryaml`) und (für die beiden JS-Dateien) Syntaxfehler (`node -c`) geprüft, aber nicht gegen einen echten `bundle exec jekyll build` getestet (Projektregel, s. Phase 0) — steht spätestens in Phase 4 an.

**Ergebnis:** Die komplette Chrome/UI-Schicht ist zweisprachig. `/de/`, `/de/problems/`, `/de/solutions/`, `/de/categories/`, `/de/landscape/` sind erreichbar, vollständig navigierbar und zeigen leere Listen (weil `_problems_de`/`_solutions_de` noch keinen Inhalt haben — das ist Phase 3).

### Phase 2 — Titel-Übersetzungstabelle (Grundlage für Phase 3) — ✅ abgeschlossen
- [x] `_data/titles_de.yml` angelegt: für alle 1013 Slugs (452 Probleme, 561 Lösungen) deutscher Titel (normale deutsche Groß-/Kleinschreibung, keine NYT-Regel), deutsche Kurzbeschreibung und ein daraus abgeleiteter deutscher Slug (`slug_de`, lowercase/Bindestriche, Umlaute transliteriert ä→ae/ö→oe/ü→ue/ß→ss), verteilt über ~17 Batches (je ~60 Einträge, alphabetisch nach EN-Slug), nach jedem Batch committet
- [x] `slug_de`- und `title_de`-Werte auf Eindeutigkeit geprüft (je einmal am Ende über die komplette Tabelle, zusätzlich nach jedem Batch); mehrere im Katalog bereits vorhandene Bedeutungsüberschneidungen im Englischen (z. B. `difficult-code-comprehension` vs. `difficult-to-understand-code`, `feature-flags` vs. `feature-toggles`, `ci-cd-pipeline` vs. `continuous-delivery`, `code-review-process-reform` vs. `code-reviews`, `static-analysis-and-linting` vs. `static-code-analysis`) mit eigens unterscheidenden deutschen Titeln aufgelöst, ohne die englischen Originale anzufassen
- [x] Vollständigkeit verifiziert: jeder EN-Slug aus `_problems/`/`_solutions/` hat genau einen Eintrag, keine überzähligen Keys
- [x] Diese Tabelle ist jetzt die einzige Quelle für Linktexte **und Ziel-Dateinamen** in Phase 3

### Phase 3 — Inhalte übersetzen (der große Teil)
Batch-weise, z. B. 25–50 Dateien pro Durchgang, mit Fortschrittstabelle unten. Pro Datei:
- [ ] Neue Datei unter `_problems_de/<slug_de>.md` bzw. `_solutions_de/<slug_de>.md` anlegen (Dateiname = `slug_de` aus der Titeltabelle, **nicht** der englische Dateiname)
- [ ] Front Matter: `title`/`description` aus der Titeltabelle übernehmen, `lang: de` setzen, `en_slug: <englischer Slug>` setzen (Rückverweis, s. Entscheidung 1/6), `category`/`related_problems`/`related_solutions`/`solutions`/`problems` unverändert als englische Slugs übernehmen (Entscheidung 5)
- [ ] Fließtext übersetzen, Abschnittsstruktur/Glyphen 1:1 beibehalten
- [ ] Jeden internen Link umschreiben: Linktext = deutscher Titel aus der Tabelle, Pfad zeigt auf `<slug_de>.md` in der `_de`-Collection (aus der Titeltabelle nachschlagen, nicht den EN-Dateinamen wiederverwenden)
- [ ] Nach jedem Batch: `python scripts/check_links.py` und `python scripts/validate_causal_links.py --detail` auf die neue Collection anwenden (Skripte müssen dafür ein Verzeichnis-Argument bekommen statt hart `_problems`/`_solutions` anzunehmen — ggf. kleiner Parametrisierungs-Task vorab)

Umfang: alle 1013 Dateien, kein Teilausschnitt (s. "Entschieden" oben). Empfehlung zur Ausführung: sobald Phase 0–2 stehen, eignet sich dieser Teil für eine pipeline-artige Bearbeitung (Datei → Übersetzen → Link-Validierung) über mehrere Agenten/Durchläufe, weil es sich um 1013 weitgehend gleichförmige, aber inhaltlich zu prüfende Einheiten handelt. Sollte der Nutzer das ausdrücklich wünschen, kann das über das Workflow-Tool orchestriert werden; das ist aber eine Ausführungsentscheidung für später, nicht Teil dieses Plans.

### Phase 4 — Qualitätssicherung
- [ ] Stichprobenartige fachliche Prüfung der Übersetzung (Anglizismen konsistent beibehalten, s. "Entschieden" oben — hier geht es nur noch um einheitliche Schreibweise, nicht um die Grundsatzfrage)
- [ ] Vollständigkeitscheck: jede `_problems/*.md` hat ein Pendant in `_problems_de/`, jede `_solutions/*.md` ein Pendant in `_solutions_de/`
- [ ] Build-Test: `bundle exec jekyll build` (voller Build, kein `--incremental`, da neue Collections/Content) + Stichproben-Navigation auf `/de/`

### Phase 5 — Laufender Betrieb
- [ ] Workflow-Regeln (`pr:generate_new_problems`, `pr:add_tech_debt`, manuelles Hinzufügen) so ergänzen, dass neue EN-Einträge eine offene ToDo-Markierung für die DE-Übersetzung bekommen (z. B. Eintrag in einer Rückstandsliste), statt dass `_problems_de`/`_solutions_de` stillschweigend veraltet
- [ ] Turnusmäßiger Nachzieh-Lauf (analog zu `calculate_related_problems.py`) für neu hinzugekommene Dateien

## Offene Fragen

Keine grundsätzlichen mehr offen — Title Case, Anglizismen, Landscape-Umfang und Vollausbau-vs.-MVP sind entschieden (s. "Entschieden" oben). Verbleibt nur ein kleiner Implementierungsdetail-Punkt, keine Entscheidung, die der Nutzer treffen muss:

- Exakte Namenskonvention für den Zusatz bei `slug_de`-Kollisionen (Phase 2) — wird beim Erzeugen der Titeltabelle pragmatisch festgelegt (z. B. unterscheidendes Wort aus dem Kontext anhängen).

## Fortschritt Phase 3 (wird bei Bearbeitung aktualisiert)

| Batch | Umfang | Status |
|---|---|---|
| 1 | Probleme: abi-compatibility-issues, accumulated-decision-debt, accumulation-of-workarounds, algorithmic-complexity-problems, alignment-and-padding-issues (5 Dateien) | ✅ |
| 2 | Probleme: analysis-paralysis, api-versioning-conflicts, approval-dependencies, architectural-mismatch, assumption-based-development, atomic-operation-overhead, authentication-bypass-vulnerabilities, author-frustration (8 Dateien) | ✅ |
| 3 | Probleme: authorization-flaws, authorization-role-explosion, automated-tooling-ineffectiveness, avoidance-behaviors, bikeshedding, blame-culture, bloated-class, bottleneck-formation (8 Dateien) | ✅ |
| 4 | Probleme: breaking-changes, brittle-codebase, budget-overruns, buffer-overflow-vulnerabilities, cache-invalidation-problems, capacity-mismatch, cargo-culting, cascade-delays (8 Dateien) | ✅ |
| 5 | Probleme: cascade-failures, change-management-chaos, changing-project-scope, circular-dependency-problems, circular-references, clever-code, code-duplication, code-review-inefficiency (8 Dateien) | ✅ |
| 6 | Probleme: cognitive-overload, communication-breakdown, communication-risk-outside-project, communication-risk-within-project, competing-priorities, competitive-disadvantage, complex-and-obscure-logic, complex-deployment-process (8 Dateien) | ✅ |
| 7 | Probleme: complex-domain-model, complex-implementation-paths, configuration-chaos, configuration-drift, conflicting-reviewer-opinions, constant-firefighting, constantly-shifting-deadlines, context-switching-overhead (8 Dateien) | ✅ |
| 8 | Probleme: convenience-driven-development, copy-paste-programming, core-modification-of-standard-software, cross-site-scripting-vulnerabilities, cross-system-data-synchronization-problems, custom-report-sprawl, customer-dissatisfaction, customization-outside-version-control (8 Dateien) | ✅ |
| 9 | Probleme: cv-driven-development, data-migration-complexities, data-migration-integrity-issues, data-protection-risk, data-structure-cache-inefficiency, database-connection-leaks, database-query-performance-issues, database-schema-design-problems (8 Dateien) | ✅ |
| 10 | Probleme: deadline-pressure, deadlock-conditions, debugging-difficulties, decision-avoidance, decision-paralysis, declining-business-metrics, defensive-coding-practices, delayed-bug-fixes (8 Dateien) | ✅ |
| 11 | Probleme: delayed-decision-making, delayed-issue-resolution, delayed-project-timelines, delayed-value-delivery, dependency-on-supplier, dependency-version-conflicts, deployment-coupling, deployment-environment-inconsistencies (8 Dateien) | ✅ |
| 12 | Probleme: deployment-risk, developer-frustration-and-burnout, development-disruption, difficult-code-comprehension, difficult-code-reuse, difficult-developer-onboarding, difficult-to-test-code, difficult-to-understand-code (8 Dateien) | ✅ |
| 13 | Probleme: difficulty-quantifying-benefits, dma-coherency-issues, duplicated-effort, duplicated-research-effort, duplicated-work, eager-to-please-stakeholders, endianness-conversion-overhead, entity-attribute-value-overuse (8 Dateien) | ✅ |
| 14 | Probleme: environment-variable-issues, error-message-information-disclosure, excessive-class-size, excessive-customization, excessive-disk-io, excessive-logging, excessive-object-allocation, extended-cycle-times (8 Dateien) | ✅ |
| 15 | Probleme: extended-research-time, extended-review-cycles, external-service-delays, false-sharing, fear-of-breaking-changes, fear-of-change, fear-of-conflict, fear-of-failure (8 Dateien) | ✅ |
| 16 | Probleme: feature-bloat, feature-creep, feature-creep-without-refactoring, feature-factory, feature-gaps, feedback-isolation, flaky-tests, frequent-changes-to-requirements (8 Dateien) | ✅ |
| 17 | Probleme: frequent-hotfixes-and-rollbacks, garbage-collection-pressure, global-state-and-side-effects, god-object-anti-pattern, gold-plating, gradual-performance-degradation, graphql-complexity-issues, growing-task-queues (8 Dateien) | ✅ |
| 18 | Probleme: hardcoded-values, hidden-dependencies, hidden-side-effects, high-api-latency, high-bug-introduction-rate, high-client-side-resource-consumption, high-connection-count, high-coupling-low-cohesion (8 Dateien) | ✅ |
| 19 | Probleme: high-database-resource-utilization, high-defect-rate-in-production, high-maintenance-costs, high-number-of-database-queries, high-resource-utilization-on-client, high-technical-debt, high-turnover, history-of-failed-changes (8 Dateien) | ✅ |
| 20 | Probleme: immature-delivery-strategy, imperative-data-fetching-logic, implementation-partner-dependency, implementation-rework, implementation-starts-without-design, implicit-knowledge, improper-event-listener-management, inability-to-innovate (8 Dateien) | ✅ |
| 21 | Probleme: inadequate-code-reviews, inadequate-configuration-management, inadequate-error-handling, inadequate-initial-reviews, inadequate-integration-tests, inadequate-mentoring-structure, inadequate-onboarding, inadequate-requirements-gathering (8 Dateien) | ✅ |
| 22 | Probleme: inadequate-test-data-management, inadequate-test-infrastructure, inappropriate-skillset, incomplete-knowledge, incomplete-projects, inconsistent-behavior, inconsistent-codebase, inconsistent-coding-standards (8 Dateien) | ✅ |
| 23 | Probleme: inconsistent-execution, inconsistent-knowledge-acquisition, inconsistent-naming-conventions, inconsistent-onboarding-experience, inconsistent-quality, incorrect-index-type, incorrect-max-connection-pool-size, increased-bug-count (8 Dateien) | ✅ |
| 24 | Probleme: increased-cognitive-load, increased-cost-of-development, increased-customer-support-load, increased-error-rates, increased-manual-testing-effort, increased-manual-work, increased-risk-of-bugs, increased-stress-and-burnout (8 Dateien) | ✅ |
| 25 | Probleme: increased-technical-shortcuts, increased-time-to-market, increasing-brittleness, index-fragmentation, individual-recognition-culture, inefficient-code, inefficient-database-indexing, inefficient-development-environment (8 Dateien) | ✅ |
| 26 | Probleme: inefficient-frontend-code, inefficient-processes, inexperienced-developers, information-decay, information-fragmentation, insecure-data-transmission, insufficient-audit-logging, insufficient-code-review (8 Dateien) | ✅ |
| 27 | Probleme: insufficient-design-skills, insufficient-testing, insufficient-worker-capacity, integer-overflow-underflow, integration-difficulties, interrupt-overhead, invisible-nature-of-technical-debt, knowledge-dependency (8 Dateien) | ✅ |
| 28 | Probleme: knowledge-gaps, knowledge-sharing-breakdown, knowledge-silos, lack-of-ownership-and-accountability, language-barriers, large-estimates-for-small-changes, large-feature-scope, large-pull-requests (8 Dateien) | ✅ |
| 29 | Probleme: large-risky-releases, lazy-loading, legacy-api-versioning-nightmare, legacy-business-logic-extraction-difficulty, legacy-code-without-tests, legacy-configuration-management-chaos, legacy-skill-shortage, legacy-system-documentation-archaeology (8 Dateien) | ✅ |
| 30 | Probleme: legal-disputes, limited-team-learning, load-balancing-problems, lock-contention, log-injection-vulnerabilities, log-spam, logging-configuration-issues, long-build-and-test-times (8 Dateien) | ✅ |
| 31+ | Restliche Probleme (alphabetisch ab „long-lived-feature-branches“ o.ä.) + alle 561 Lösungen | offen |

Stand: 109 / 452 Problem-Dateien (_problems_de/), 0 / 561 Lösungs-Dateien (_solutions_de/). Jeder Batch wird direkt committet und nach `claude/german-translation-plan-wsmx5o` gepusht. Manche Links in bereits übersetzten Dateien zeigen noch auf `slug_de`-Ziele, die erst in einem späteren Batch entstehen — das ist erwartet und löst sich auf, sobald der komplette Katalog übersetzt ist; `check_links.py`/`validate_causal_links.py --detail` sollten erst nach Abschluss von Phase 3 (oder mit `--dry-run`-artiger Toleranz für "noch nicht übersetzt") gegen `_problems_de`/`_solutions_de` laufen.
