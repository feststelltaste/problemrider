---
title: Unzureichende Erst-Reviews
description: Code-Reviews der ersten Runde sind unvollständig oder oberflächlich
  und versäumen es, wichtige Probleme zu identifizieren, die erst in späteren Review-Runden
  entdeckt werden.
category:
- Code
- Process
related_problems:
- slug: inadequate-code-reviews
  similarity: 0.8
- slug: insufficient-code-review
  similarity: 0.8
- slug: reviewer-inexperience
  similarity: 0.75
- slug: superficial-code-reviews
  similarity: 0.75
- slug: code-review-inefficiency
  similarity: 0.75
- slug: review-process-breakdown
  similarity: 0.75
solutions:
- code-review-process-reform
- code-review-guidelines
- checklists
- static-analysis-and-linting
- code-quality-gates
- small-change-batches
- definition-of-done
- lightweight-design-review
- pair-and-mob-programming
- team-working-agreements
layout: problem
lang: de
en_slug: inadequate-initial-reviews
---

## Description

Unzureichende Erst-Reviews treten auf, wenn die erste Runde des Code-Reviews es versäumt, wichtige Probleme, Designfehler oder mögliche Verbesserungen zu identifizieren, die früh hätten erfasst werden sollen. Dies resultiert in mehreren Review-Zyklen, in denen in jeder Runde neue Probleme entdeckt werden, was den Review-Prozess unnötig verlängert und Frustration sowohl für Autoren als auch für Reviewer erzeugt. Das Problem zeigt an, dass Reviewer während ihrer ersten Untersuchung des Codes keine gründliche Analyse durchführen.

## Indicators ⟡

- Probleme, die offensichtlich hätten sein sollen, werden erst in späteren Review-Runden identifiziert
- Jede Review-Runde deckt völlig neue Kategorien von Problemen auf
- Reviewer geben zunächst nur oberflächliches Feedback, dann tiefere Analyse in nachfolgenden Runden
- Wichtige Design- oder architektonische Probleme werden übersehen, bis Implementierungsdetails reviewt wurden
- Die Review-Qualität verbessert sich in späteren Runden erheblich im Vergleich zu Erst-Reviews

## Symptoms ▲

- [Verlängerte Review-Zyklen](verlaengerte-review-zyklen.md)
<br/>  In Erst-Reviews übersehene Probleme erzwingen mehrere Review-Runden, was die Zeit von der Einreichung bis zur Genehmigung erheblich verlängert.
- [Frustration der Autoren](frustration-der-autoren.md)
<br/>  Entwickler werden frustriert, wenn in jeder Review-Runde neue Probleme auftauchen, die anfänglich hätten erfasst werden sollen.
- [Langsame Entwicklungsgeschwindigkeit](langsame-entwicklungsgeschwindigkeit.md)
<br/>  Mehrere Review-Zyklen verzögern Code-Merges und -Lieferung, was den Gesamtdurchsatz des Teams verringert.
- [Ineffizienz im Code-Review](ineffizienz-im-code-review.md)
<br/>  Der Review-Prozess verschwendet Zeit, während Probleme, die einmal erfasst werden sollten, mehrere Durchgänge erfordern, um vollständig identifiziert zu werden.

## Causes ▼

- [Unzureichende Code-Reviews](unzureichende-code-reviews.md)
<br/>  Eine generelle Kultur unzureichender Code-Reviews setzt das Muster für oberflächliche Erst-Reviews, bei denen kritische Probleme übersehen werden.
- [Unerfahrenheit der Reviewer](unerfahrenheit-der-reviewer.md)
<br/>  Unerfahrene Reviewer konzentrieren sich auf oberflächliche Probleme wie Formatierung, weil ihnen die Expertise fehlt, tiefere Probleme zu identifizieren.
- [Zeitdruck](zeitdruck.md)
<br/>  Reviewer unter Zeitdruck überfliegen Code, statt gründliche Erstanalyse durchzuführen.
- [Große Pull Requests](grosse-pull-requests.md)
<br/>  Übergroße Pull Requests machen es kognitiv schwierig, alle Probleme in einem einzigen Durchgang zu identifizieren.
- [Nitpicking-Kultur](nitpicking-kultur.md)
<br/>  Eine auf triviale Stilfragen fokussierte Kultur trainiert Reviewer darauf, auf oberflächliche Details statt auf substanzielle Design- und Logikprobleme zu achten.

## Detection Methods ○

- **Analyse des Problem-Entdeckungsmusters:** Nachverfolgung, wann unterschiedliche Arten von Problemen über Review-Runden hinweg identifiziert werden
- **Messung der Erstrunden-Wirksamkeit:** Bewertung, welcher Prozentsatz der Gesamtprobleme in Erst-Reviews erfasst wird
- **Progression der Review-Qualität:** Analyse, ob Review-Feedback in späteren Runden erheblich tiefer wird
- **Bewertung der Reviewer-Leistung:** Vergleich der Fähigkeit unterschiedlicher Reviewer, Probleme früh zu identifizieren
- **Korrelation des Zeitinvestments:** Untersuchung des Zusammenhangs zwischen für Erst-Review aufgewendeter Zeit und Problem-Entdeckung

## Examples

Ein Entwickler reicht eine komplexe Feature-Implementierung ein und erhält anfängliches Review-Feedback, das sich vollständig auf Code-Formatierung und Variablenbenennung konzentriert. Erst in der dritten Review-Runde bemerkt ein Reviewer, dass der Algorithmus O(n²)-Komplexität hat und optimiert werden könnte, und in der vierten Runde identifiziert jemand, dass die Fehlerbehandlung Datenverfälschung verursachen könnte. Die Probleme, die die bedeutendste Nacharbeit erforderten, hätten sofort erfasst werden sollen, wurden aber übersehen, weil der erste Reviewer nur oberflächliche Stilfragen betrachtete. Ein weiteres Beispiel betrifft ein sicherheitskritisches Authentifizierungs-Feature, bei dem der erste Reviewer die Implementierung genehmigt, nachdem er nur die Happy-Path-Logik geprüft hat. In der zweiten Runde identifiziert ein anderer Reviewer, dass die Fehlerbehandlung sensible Informationen preisgibt, und in der dritten Runde entdeckt jemand, dass das Session-Management eine Race-Condition-Schwachstelle hat. Diese kritischen Sicherheitsprobleme hätten der primäre Fokus des Erst-Reviews sein sollen, wurden aber aufgrund oberflächlicher Analyse übersehen.
