---
title: Prokrastination bei komplexen Aufgaben
description: Schwierige oder kognitiv anspruchsvolle Arbeit wird konsequent zugunsten
  einfacherer, unmittelbar befriedigenderer Aufgaben aufgeschoben.
category:
- Culture
- Process
related_problems:
- slug: avoidance-behaviors
  similarity: 0.85
- slug: decision-avoidance
  similarity: 0.65
- slug: delayed-decision-making
  similarity: 0.6
- slug: accumulated-decision-debt
  similarity: 0.6
- slug: work-blocking
  similarity: 0.6
- slug: increased-cognitive-load
  similarity: 0.6
solutions:
- iterative-development
- mikado-method
- small-change-batches
- technical-spike
- pair-and-mob-programming
- work-in-progress-limits
- walking-skeleton
- preparatory-refactoring
- code-reading-sessions
- psychological-safety-practices
layout: problem
lang: de
en_slug: procrastination-on-complex-tasks
---

## Description

Prokrastination bei komplexen Aufgaben tritt auf, wenn Entwickler konsequent den Beginn schwieriger, kognitiv anspruchsvoller oder unsicherer Arbeit verzögern oder vermeiden, zugunsten einfacherer, unmittelbar befriedigenderer Aktivitäten. Dieses Verhalten entsteht oft aus psychologischen Faktoren wie Angst vor Scheitern, Perfektionismus oder kognitiver Überlastung. Während ein gewisses Maß an Aufgabenpräferenz natürlich ist, kann systematische Prokrastination bei komplexer Arbeit zur Anhäufung schwieriger Probleme und erhöhten technischen Schulden führen.

## Indicators ⟡

- Schwierige Aufgaben bleiben unbegonnen, während einfachere Aufgaben schnell abgeschlossen werden
- Teammitglieder finden Gründe, an anderen Aktivitäten zu arbeiten, wenn ihnen komplexe Probleme zugewiesen werden
- Komplexe Features rutschen konsequent zu späteren Sprints oder Iterationen
- Entwickler äußern Angst oder Stress, wenn sie über herausfordernde Arbeit diskutieren
- Einfache Fehler werden sofort behoben, während komplexe Probleme im Backlog verbleiben

## Symptoms ▲

- [Anhäufung von Workarounds](anhaeufung-von-workarounds.md)
<br/>  Statt die schwierige Korrektur anzugehen, erstellen Entwickler Workarounds, die dem System Komplexität hinzufügen.
- [Verzögerte Problemlösung](verzoegerte-problemloesung.md)
<br/>  Komplexe Probleme bleiben monatelang unbearbeitet im Backlog, weil sie konsequent aufgeschoben werden.
- [Zunehmende Brüchigkeit](zunehmende-bruechigkeit.md)
<br/>  Aufgeschobene architektonische Arbeit macht das System brüchiger, während sich Änderungen um problematische Bereiche herum anhäufen.
- [Zeitdruck](zeitdruck.md)
<br/>  Aufgeschobene komplexe Arbeit wird schließlich dringend, was Termindruck in letzter Minute schafft.

## Causes ▼

- [Kognitive Überlastung](kognitive-ueberlastung.md)
<br/>  Mentale Erschöpfung durch Systemkomplexität lässt Entwickler die zusätzliche kognitive Last schwieriger Aufgaben vermeiden.
- [Schwer verständliche Codebasis](schwer-verstaendliche-codebasis.md)
<br/>  Wenn Code schwer zu verstehen ist, erhöht sich die wahrgenommene Schwierigkeit komplexer Aufgaben, was Vermeidung fördert.

## Detection Methods ○

- **Analyse der Aufgabenbeginn-Verzögerung:** Messung der Zeit zwischen Aufgabenzuweisung und tatsächlichem Arbeitsbeginn
- **Komplexität vs. Abschlusszeit:** Vergleich von Komplexitätsbewertungen mit tatsächlichen Abschlussmustern
- **Backlog-Alter nach Komplexität:** Nachverfolgung, wie lange komplexe versus einfache Aufgaben im Backlog verbleiben
- **Entwickler-Feedback-Befragungen:** Befragung zu Faktoren, die Aufgabenwahl und -vermeidung beeinflussen
- **Sprint-Planungsverhalten:** Beobachtung, für welche Aufgaben sich freiwillig gemeldet wird versus welche während der Planung vermieden werden

## Examples

Ein Entwicklungsteam hat drei architektonische Refactoring-Aufgaben in seinem Backlog, die seit vier Monaten dort sind, während im selben Zeitraum Dutzende kleinerer Feature-Ergänzungen und Fehlerbehebungen abgeschlossen wurden. Teammitglieder melden sich während der Sprint-Planung konsequent freiwillig für die kleineren Aufgaben und finden Gründe, warum die Refactoring-Arbeit „noch nicht ganz bereit" ist oder „mehr Analyse benötigt". Das vermiedene Refactoring wird zunehmend dringender, während das System schwerer zu warten wird, aber bis es angegangen werden muss, ist die Arbeit aufgrund von Änderungen um die problematischen Bereiche herum noch komplexer und riskanter geworden. Ein weiteres Beispiel betrifft einen Entwickler, der einen komplexen Algorithmus für Datenverarbeitung implementieren muss, aber immer wieder andere Aufgaben findet, an denen er zuerst arbeitet – Dokumentation aktualisieren, kleinere UI-Probleme beheben, Datenbankabfragen optimieren. Die Algorithmus-Implementierung bleibt wochenlang unangetastet, während der Entwickler mit weniger herausfordernder Arbeit beschäftigt bleibt, was schließlich dazu führt, dass das Feature seinen Termin verpasst und Notfall-Wochenendarbeit zur Fertigstellung erfordert.
