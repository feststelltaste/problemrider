---
title: Verzögerte Entscheidungsfindung
description: Wichtige Entscheidungen, die den Entwicklungsfortschritt betreffen,
  werden aufgeschoben oder dauern übermäßig lange, was Engpässe und Unsicherheit
  erzeugt.
category:
- Management
- Process
- Team
related_problems:
- slug: decision-avoidance
  similarity: 0.85
- slug: accumulated-decision-debt
  similarity: 0.75
- slug: decision-paralysis
  similarity: 0.75
- slug: work-blocking
  similarity: 0.75
- slug: approval-dependencies
  similarity: 0.7
- slug: analysis-paralysis
  similarity: 0.7
solutions:
- architecture-decision-records
- decision-rights-and-escalation
- technical-spike
- explicit-prioritization-framework
- clear-ownership-model
- architecture-review-board
- team-retrospectives
- written-first-communication
- lightweight-design-review
- pilot-projects
- cost-of-delay
- no-regret-moves
layout: problem
lang: de
en_slug: delayed-decision-making
---

## Description

Verzögerte Entscheidungsfindung entsteht, wenn wichtige Entscheidungen, die die Entwicklungsarbeit betreffen, aufgeschoben werden, übermäßig lange dauern oder in Genehmigungsprozessen stecken bleiben. Diese Verzögerung schafft Unsicherheit für Teammitglieder, blockiert den Fortschritt bei abhängiger Arbeit und kann zu verpassten Gelegenheiten oder suboptimalen Ergebnissen führen, wenn Entscheidungen schließlich unter Zeitdruck getroffen werden. Das Problem entspringt oft unklarer Entscheidungsautorität, Angst vor falschen Entscheidungen oder übermäßig komplexen Genehmigungsprozessen.

## Indicators ⟡

- Entwicklungsarbeit ist häufig blockiert und wartet auf Entscheidungen
- Dieselben Entscheidungen werden wiederholt diskutiert, ohne gelöst zu werden
- Entscheidungsträger verlangen übermäßige Analyse, bevor sie Entscheidungen treffen
- Wichtige Entscheidungen werden in letzter Minute unter Druck getroffen
- Teammitglieder sind sich unklar darüber, wer die Autorität hat, bestimmte Arten von Entscheidungen zu treffen

## Symptoms ▲

- [Arbeitsblockade](arbeitsblockade.md)
<br/>  Entwicklungsaufgaben, die von ungetroffenen Entscheidungen abhängen, können nicht fortschreiten, was Engpässe im Workflow erzeugt.
- [Verzögerte Projektzeitpläne](verzoegerte-projektzeitplaene.md)
<br/>  Projekte geraten in Zeitverzug, während die Umsetzungsarbeit stillsteht und auf zu treffende Entscheidungen wartet.
- [Ressourcenverschwendung](ressourcenverschwendung.md)
<br/>  Teams verbringen Zeit mit dem Bau von Wegwerf-Prototypen, der Teilnahme an sich wiederholenden Meetings und der Recherche von Optionen, die möglicherweise nie ausgewählt werden.
- [Angehäufte Entscheidungsschulden](angehaeufte-entscheidungsschulden.md)
<br/>  Aufgeschobene Entscheidungen häufen sich an und werden voneinander abhängig, was sie zunehmend schwerer lösbar macht.
- [Frustration der Stakeholder](frustration-der-stakeholder.md)
<br/>  Stakeholder werden frustriert, wenn der Projektfortschritt sichtbar aufgrund ungelöster Entscheidungen stagniert.

## Causes ▼

- [Freigabe-Abhängigkeiten](freigabe-abhaengigkeiten.md)
<br/>  Entscheidungen, die die Freigabe bestimmter Personen erfordern, stecken fest, wenn diese Personen nicht verfügbar oder überlastet sind.
- [Analyse-Lähmung](analyse-laehmung.md)
<br/>  Teams bleiben in endloser Recherche und Bewertung von Optionen stecken, ohne sich auf eine Wahl festzulegen.
- [Mikromanagement-Kultur](mikromanagement-kultur.md)
<br/>  Eine Kultur, die Management-Genehmigung für Routine-technische Entscheidungen verlangt, erzeugt Verzögerungen, während Entscheidungen sich zur Überprüfung anstauen.
- [Schuldzuweisungskultur](schuldzuweisungskultur.md)
<br/>  Die Angst, für falsche Entscheidungen verantwortlich gemacht zu werden, führt dazu, dass Entscheidungsträger Entscheidungen verzögern, bis sie sich völlig sicher fühlen.

## Detection Methods ○

- **Entscheidungs-Tracking:** Beobachtung, wie lange unterschiedliche Arten von Entscheidungen von der Identifikation bis zur Lösung brauchen
- **Analyse blockierter Arbeit:** Nachverfolgung, wie oft Entwicklungsarbeit blockiert ist und auf Entscheidungen wartet
- **Bewertung des Entscheidungsrückstaus:** Identifikation ausstehender Entscheidungen und ihrer Auswirkung auf den Projektfortschritt
- **Stakeholder-Feedback:** Sammlung von Eingaben zur Wirksamkeit der Entscheidungsfindung von Teammitgliedern
- **Entscheidungsqualitäts-Review:** Bewertung, ob verzögerte Entscheidungen tatsächlich zu besseren Ergebnissen führen

## Examples

Ein Entwicklungsteam muss zwischen zwei unterschiedlichen Datenbanktechnologien für ein neues Feature wählen, aber das Management diskutiert die Entscheidung seit sechs Wochen, ohne zu einem Schluss zu kommen. In der Zwischenzeit kann das Entwicklungsteam mit der Umsetzung nicht fortfahren, weil die Datenbankwahl die gesamte Architektur betrifft. Teammitglieder verbringen Zeit damit, beide Optionen wiederholt zu recherchieren, Prototypen zu erstellen, die möglicherweise nicht genutzt werden, und an mehreren Meetings teilzunehmen, die nicht zu Entscheidungen führen. Letztlich wird die Entscheidung überstürzt getroffen, um einen Termin einzuhalten, ohne ordentliche Berücksichtigung der durchgeführten Recherche. Ein weiteres Beispiel betrifft eine API-Design-Entscheidung, bei der das Team zwischen REST- und GraphQL-Ansätzen wählen muss. Die Entscheidung wird durch mehrere Management-Ebenen eskaliert, wobei jede Ebene zusätzliche Analyse und Dokumentation verlangt. Drei Monate später, als die Entscheidung schließlich getroffen wird, haben sich die Geschäftsanforderungen geändert, und die ursprüngliche Analyse ist nicht mehr relevant.
