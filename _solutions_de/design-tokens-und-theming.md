---
title: Design Tokens und Theming
description: Plattformunabhängige Kodierung visueller Design-Entscheidungen für Theming
  und Konsistenz.
category:
- Architecture
- Code
quality_tactics_url: https://qualitytactics.de/en/usability/design-tokens/
problems:
- inconsistent-codebase
- inconsistent-behavior
- poor-user-experience-ux-design
- high-maintenance-costs
- maintenance-overhead
- technology-stack-fragmentation
- difficult-code-reuse
layout: solution
lang: de
en_slug: design-tokens
related_solutions:
- slug: style-guide
  similarity: 0.75
- slug: consistent-user-interface
  similarity: 0.7
- slug: consistent-terminology
  similarity: 0.65
- slug: visual-hierarchy
  similarity: 0.65
- slug: pattern-language
  similarity: 0.65
- slug: user-centered-design
  similarity: 0.65
---

## Description

Design Tokens extrahieren visuelle Design-Entscheidungen — Farben, Abstände, Typografie, Radien — aus verstreuten fest codierten Werten in CSS und Stylesheets in eine einzige, plattformunabhängige Quelle der Wahrheit, die das tatsächliche Styling für jede genutzte Technologie erzeugt. Legacy-Systeme, die über Jahre separater Frontend-Bemühungen gewachsen sind, manchmal über mehrere Frameworks hinweg, enden typischerweise mit mehreren leicht unterschiedlichen Schattierungen derselben Markenfarbe und inkonsistenten Abständen, die niemand beabsichtigt hat, einfach weil es nie einen Ort gab, an dem diese Entscheidungen lebten. Schrittweise zu Tokens zu migrieren, fest codierte Werte zu ersetzen, während Dateien während normaler Wartung angefasst werden, statt in einer großen Neuschreibung, lässt eine systemweite visuelle Änderung oder ein neues Theme zu einem Token-Update werden, statt zu einer Suchen-und-Ersetzen-Aktion über Tausende Zeilen Legacy-Styling.

## How to Apply ◆

> Legacy-Systeme haben oft visuelle Design-Entscheidungen als fest codierte Werte über die Codebasis verstreut. Design Tokens zentralisieren diese Entscheidungen in einer einzigen Quelle der Wahrheit, die über Plattformen hinweg angewandt werden kann.

- Extrahieren Sie fest codierte Farbwerte, Schriftgrößen, Abstandseinheiten und Rahmenradien aus Legacy-CSS, Stylesheets und Inline-Styles in eine zentralisierte Token-Datei mittels eines Formats wie JSON oder YAML.
- Definieren Sie eine Token-Benennungshierarchie, die primitive Tokens (rohe Werte wie spezifische Hex-Farben) von semantischen Tokens (zweckbasierte Namen wie „color-error" oder „spacing-section") trennt, um das System pflegbar und bedeutungsvoll zu machen.
- Implementieren Sie eine Build-Pipeline, die Design Tokens in plattformspezifische Formate transformiert: CSS-Custom-Properties für Web, Ressourcendateien für Mobile und Konstanten für Desktop-Anwendungen.
- Wenden Sie Tokens schrittweise während der Legacy-Wartung an. Wenn Sie eine Datei mit fest codierten visuellen Werten anfassen, ersetzen Sie sie durch Token-Referenzen, statt einen vollständigen systemweiten Ersatz auf einmal zu versuchen.
- Unterstützen Sie Theming, indem Sie semantische Tokens auf unterschiedliche Wertesätze für Light Mode, Dark Mode, Hochkontrastmodus und markenspezifische Varianten abbilden.
- Dokumentieren Sie das Token-System mit visuellen Beispielen, sodass Entwickler das korrekte Token nachschlagen können, statt zu raten oder neue fest codierte Werte einzuführen.

## Tradeoffs ⇄

> Design Tokens schaffen eine mächtige Abstraktion für visuelle Konsistenz, erfordern aber Disziplin und Tooling, um sie effektiv zu verwalten.

**Vorteile:**

- Ermöglicht systemweite visuelle Änderungen durch ein einziges Token-Update, statt Tausende Zeilen Legacy-CSS nach fest codierten Werten zu durchsuchen.
- Unterstützt Theming und Barrierefreiheitsmodi, ohne Stylesheets zu duplizieren oder parallele visuelle Codebasen zu pflegen.
- Stellt visuelle Konsistenz über unterschiedliche Technologie-Stacks innerhalb des Legacy-Systems sicher, selbst wenn unterschiedliche Module unterschiedliche Frontend-Frameworks nutzen.
- Reduziert Wartungsoverhead, indem redundante Farb- und Abstandsdefinitionen, die über die Codebasis verstreut sind, eliminiert werden.

**Kosten und Risiken:**

- Die Einrichtung der Token-Infrastruktur und Build-Pipeline erfordert anfängliche Investition, bevor irgendein visueller Nutzen realisiert wird.
- Die Migration einer großen Legacy-Codebasis von fest codierten Werten zu Tokens ist mühsam und muss schrittweise erfolgen, um Regressionen zu vermeiden.
- Overengineering der Token-Hierarchie mit zu vielen Abstraktionsschichten kann das System schwerer verständlich machen als die fest codierten Werte, die es ersetzte.
- Teams müssen die Disziplin annehmen, Tokens für alle neue Arbeit zu nutzen, sonst kehrt das System schrittweise zur Inkonsistenz zurück.

## How It Could Be

> Legacy-Systeme mit langer Historie häufen Dutzende leicht unterschiedlicher Schattierungen derselben Farbe und inkonsistente Abstände an, was visuelles Rauschen erzeugt, das Professionalität untergräbt.

Eine über fünfzehn Jahre gebaute Legacy-Unternehmensanwendung hat ihre UI über drei Frontend-Technologien verteilt: JSP-Seiten, ein vor fünf Jahren hinzugefügtes React-basiertes Modul und ein kürzlich hinzugefügtes Angular-Dashboard. Jedes nutzt seine eigenen Farbdefinitionen, was zu drei unterschiedlichen Schattierungen des primären Blaus des Unternehmens und inkonsistenten Abständen überall führt. Das Team extrahiert alle Farb- und Abstandswerte aus allen drei Codebasen in eine gemeinsame Design-Token-Datei. Ein Build-Schritt erzeugt CSS-Custom-Properties für die JSP-Seiten, ein JavaScript-Modul für React und eine TypeScript-Konstantendatei für Angular. Nach drei Monaten schrittweiser Migration während routinemäßiger Wartung erreicht die Anwendung zum ersten Mal visuelle Konsistenz über alle Module hinweg, und die Implementierung eines Dark Mode wird zu einer Frage der Erstellung eines alternativen Token-Wertesatzes, statt Hunderte von CSS-Regeln neu zu schreiben.
