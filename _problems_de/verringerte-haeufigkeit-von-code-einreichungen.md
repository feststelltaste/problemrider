---
title: Verringerte Häufigkeit von Code-Einreichungen
description: Entwickler bündeln mehrere Änderungen oder verzögern Einreichungen,
  um häufige Code-Review-Zyklen zu vermeiden, was Feedback-Qualität und Integrationshäufigkeit
  verringert.
category:
- Process
- Team
related_problems:
- slug: reduced-review-participation
  similarity: 0.75
- slug: large-pull-requests
  similarity: 0.75
- slug: extended-review-cycles
  similarity: 0.75
- slug: author-frustration
  similarity: 0.7
- slug: code-review-inefficiency
  similarity: 0.7
- slug: inadequate-code-reviews
  similarity: 0.7
solutions:
- development-environment-optimization
- development-workflow-automation
- regression-testing
- small-change-batches
- code-review-guidelines
- trunk-based-development
- continuous-integration
- work-in-progress-limits
- team-retrospectives
- fast-feedback-loops
layout: problem
lang: de
en_slug: reduced-code-submission-frequency
---

## Description

Verringerte Häufigkeit von Code-Einreichungen tritt auf, wenn Entwickler absichtlich mehrere Änderungen bündeln oder das Einreichen von Code zum Review verzögern, um den Overhead und die Frustration häufiger Review-Zyklen zu vermeiden. Während dies aus individueller Perspektive effizient erscheinen mag, führt es zu größeren, komplexeren Änderungen, die schwerer effektiv zu reviewen sind, erhöht Integrationsrisiken und verringert die kollaborativen Vorteile häufigen Feedbacks.

## Indicators ⟡

- Entwickler reichen große Pull Requests mit mehreren nicht zusammenhängenden Änderungen ein
- Tage oder Wochen vergehen zwischen Code-Einreichungen aktiver Entwickler
- Teammitglieder erwähnen, zu warten, um „alles fertigzustellen", bevor sie zum Review einreichen
- Pull-Request-Größen sind konsequent größer als von Team-Richtlinien empfohlen
- Entwickler äußern Zurückhaltung, unfertige Arbeit oder inkrementelle Änderungen einzureichen

## Symptoms ▲

- [Große Pull Requests](grosse-pull-requests.md)
<br/>  Das Bündeln mehrerer Änderungen, um häufige Review-Zyklen zu vermeiden, produziert direkt übergroße Pull Requests.
- [Unzureichende Code-Reviews](unzureichende-code-reviews.md)
<br/>  Große gebündelte Einreichungen sind schwerer gründlich zu reviewen, was die Qualität und Effektivität von Code-Reviews verringert.
- [Kein kontinuierlicher Feedback-Loop](kein-kontinuierlicher-feedback-loop.md)
<br/>  Seltenere Einreichungen bedeuten, dass Entwickler später Feedback erhalten, wenn Design-Entscheidungen schwerer zu ändern sind.

## Causes ▼

- [Review-Engpässe](review-engpaesse.md)
<br/>  Langsame Review-Prozesse entmutigen häufige Einreichungen, während Entwickler es vermeiden, wiederholt auf Reviews zu warten.
- [Vermeidung des Review-Prozesses](vermeidung-des-review-prozesses.md)
<br/>  Frustration mit dem Review-Prozess führt dazu, dass Entwickler ihre Exposition dazu minimieren, indem sie Änderungen bündeln.
- [Angst vor Veränderung](angst-vor-veraenderung.md)
<br/>  Angst, Code einzureichen, der kritisiert werden könnte, verursacht, dass Entwickler Einreichungen verzögern, bis sie das Gefühl haben, alles sei perfekt.
- [Ineffizienz im Code-Review](ineffizienz-im-code-review.md)
<br/>  Ineffiziente Review-Prozesse, die sich auf triviale Probleme fokussieren, entmutigen Entwickler davon, häufig einzureichen.

## Detection Methods ○

- **Einreichungshäufigkeitsverfolgung:** Überwachung, wie oft einzelne Entwickler Code zum Review einreichen
- **Pull-Request-Größenanalyse:** Nachverfolgung von Größe und Komplexität von Code-Einreichungen über die Zeit
- **Entwicklerverhaltensbefragungen:** Sammlung von Feedback zu Gründen für das Bündeln von Änderungen oder Verzögern von Einreichungen
- **Integrationshäufigkeitsmessung:** Bewertung, wie oft Code in Hauptbranches integriert wird
- **Kollaborationsmusteranalyse:** Bewertung, ob verringerte Einreichungen mit verringerter Team-Zusammenarbeit korrelieren

## Examples

Ein Entwickler, der an einem neuen Feature arbeitet, wird frustriert, nachdem sein erster kleiner Pull Request vier Review-Runden mit umfangreichen Stildebatten durchläuft. Für seine nächste Änderung entscheidet er, das gesamte Feature zu implementieren, alle Tests zu schreiben, Dokumentation zu aktualisieren und drei zugehörige Fehlerbehebungen zu handhaben, bevor er irgendetwas zum Review einreicht. Der resultierende 800-Zeilen-Pull-Request ist für Reviewer schwer umfassend zu analysieren, enthält mehrere nicht zusammenhängende Änderungen, die separat bewertet werden sollten, und braucht zwei Wochen zum Review statt der wenigen Tage, die jede einzelne Änderung erfordert hätte. Ein weiteres Beispiel betrifft ein Teammitglied, das aufhört, täglichen Fortschritt einzureichen, weil sich vorherige Reviews stark auf kleinere Formatierungsprobleme fokussierten. Sie beginnen, eine Woche am Stück zu arbeiten, bevor sie einreichen, was Integrationskonflikte mit der Arbeit von Teamkollegen schafft und es dem Team erschwert, frühes Feedback zu Design-Entscheidungen zu geben, die später im Entwicklungsprozess schwer zu ändern sind.
