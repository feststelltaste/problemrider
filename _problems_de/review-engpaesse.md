---
title: Review-Engpässe
description: Der Code-Review-Prozess wird zu einem erheblichen Engpass, der die Auslieferung
  neuer Features und Bugfixes verzögert.
category:
- Process
- Team
related_problems:
- slug: code-review-inefficiency
  similarity: 0.75
- slug: bottleneck-formation
  similarity: 0.7
- slug: review-process-breakdown
  similarity: 0.7
- slug: inadequate-code-reviews
  similarity: 0.7
- slug: extended-review-cycles
  similarity: 0.7
- slug: maintenance-bottlenecks
  similarity: 0.7
solutions:
- code-review-process-reform
- small-change-batches
- work-in-progress-limits
- code-review-guidelines
- pair-and-mob-programming
- trunk-based-development
- clear-roles-and-ownership
- team-retrospectives
- fast-feedback-loops
- delivery-performance-metrics
layout: problem
lang: de
en_slug: review-bottlenecks
---

## Description
Review-Engpässe treten auf, wenn der Code-Review-Prozess konsequent den Entwicklungszyklus verlangsamt. Dies kann aus verschiedenen Gründen geschehen, etwa zu wenige Reviewer, große und komplexe Pull Requests oder eine Kultur, in der Reviews nicht priorisiert werden. Wenn Code-Reviews zum Engpass werden, führt dies zu Frustration bei Entwicklern, verzögerten Releases und einem Rückgang der gesamten Entwicklungsgeschwindigkeit.

## Indicators ⟡
- Pull Requests liegen lange, ohne reviewt zu werden.
- Entwickler wechseln häufig den Kontext, während sie auf Reviews warten.
- Das Team hat eine niedrige Deployment-Frequenz.
- Es gibt viel Druck, Pull Requests schnell zu genehmigen, selbst wenn sie nicht bereit sind.

## Symptoms ▲

- [Langsame Entwicklungsgeschwindigkeit](langsame-entwicklungsgeschwindigkeit.md)
<br/>  Wenn Code-Reviews die Entwicklungs-Pipeline blockieren, sinkt die Gesamtgeschwindigkeit des Teams, während Entwickler auf Genehmigungen warten.
- [Übereilte Genehmigungen](uebereilte-genehmigungen.md)
<br/>  Der Druck, den Review-Rückstand abzubauen, verleitet Reviewer dazu, Änderungen hastig zu genehmigen, ohne gründliche Prüfung.
- [Vermeidung des Review-Prozesses](vermeidung-des-review-prozesses.md)
<br/>  Frustration über lange Wartezeiten bei Reviews motiviert Entwickler, Wege zu finden, den Review-Prozess ganz zu umgehen.
- [Verzögerte Wertlieferung](verzoegerte-wertlieferung.md)
<br/>  Features und Fixes, die fertig sind, aber in Review-Warteschlangen feststecken, erreichen Nutzer nicht, was die Wertlieferung verzögert.
- [Frustration der Autoren](frustration-der-autoren.md)
<br/>  Entwickler werden frustriert, wenn ihre abgeschlossene Arbeit über längere Zeit untätig in Review-Warteschlangen liegt.

## Causes ▼

- [Verringerte Review-Beteiligung](verringerte-review-beteiligung.md)
<br/>  Wenn nur wenige Teammitglieder an Reviews teilnehmen, fällt alle Review-Arbeit auf eine kleine Zahl von Personen, was einen Engpass schafft.
- [Große Pull Requests](grosse-pull-requests.md)
<br/>  Große Pull Requests brauchen viel länger, um gründlich reviewt zu werden, was mehr Reviewer-Zeit verbraucht und Rückstände schafft.
- [Freigabe-Abhängigkeiten](freigabe-abhaengigkeiten.md)
<br/>  Anforderungen, dass bestimmte Personen Änderungen genehmigen müssen, schaffen Engpässe, wenn diese Personen nicht verfügbar sind.
- [Zeitdruck](zeitdruck.md)
<br/>  Unter Zeitdruck priorisieren Entwickler ihre eigenen Aufgaben über das Reviewen fremden Codes, was Review-Warteschlangen wachsen lässt.

## Detection Methods ○
- **Pull-Request-Durchlaufzeit:** Verfolgung der Zeit von der Erstellung eines Pull Requests bis zum Merge.
- **Reviewer-Auslastung:** Analyse der Anzahl von Pull Requests, die jedem Reviewer zugewiesen sind.
- **Entwicklerbefragungen:** Befragung von Entwicklern zu ihrer Erfahrung mit dem Code-Review-Prozess und ob sie ihn als Engpass empfinden.

## Examples
Ein Team hat die Regel, dass alle Pull Requests von zwei Personen reviewt werden müssen. Es gibt jedoch nur zwei erfahrene Entwickler im Team, die qualifiziert sind, Code zu reviewen. Infolgedessen liegen Pull Requests oft tagelang oder sogar wochenlang, bevor sie reviewt werden. Dies verursacht viel Frustration bei den Junior-Entwicklern, die ihren Code nicht rechtzeitig gemerged bekommen. In einem weiteren Beispiel hat ein Team eine Kultur, in der Code-Reviews nicht priorisiert werden. Von Entwicklern wird erwartet, ihre eigene Arbeit abzuschließen, bevor sie den Code anderer reviewen. Dies führt zu einer Situation, in der Pull Requests oft lange liegen, bevor sie reviewt werden, was den gesamten Entwicklungsprozess verlangsamt.
