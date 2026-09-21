---
title: Scope Creep
description: Projektanforderungen wachsen kontinuierlich ohne ordentliche Kontrolle
  oder Auswirkungsanalyse, was Zeitpläne, Budgets und die ursprünglichen Ziele gefährdet.
category:
- Process
- Requirements
related_problems:
- slug: feature-creep
  similarity: 0.75
- slug: no-formal-change-control-process
  similarity: 0.75
- slug: frequent-changes-to-requirements
  similarity: 0.7
- slug: changing-project-scope
  similarity: 0.7
- slug: feature-creep-without-refactoring
  similarity: 0.65
- slug: unrealistic-deadlines
  similarity: 0.65
solutions:
- change-management-process
- evolutionary-requirements-development
- formal-change-control-process
- product-owner
- requirements-analysis
- stakeholder-feedback-loops
- story-mapping
- definition-of-ready
layout: problem
lang: de
en_slug: scope-creep
---

## Description

Scope Creep ist die schleichende Ausdehnung der Ziele und Liefergegenstände eines Projekts über das ursprünglich Geplante hinaus, ohne eine entsprechende Anpassung von Zeit, Budget oder Ressourcen. Es geschieht oft graduell, durch eine Reihe scheinbar kleiner Ergänzungen oder Änderungen, die das Projekt über die Zeit erheblich aufblähen. Dieses unkontrollierte Wachstum kann Zeitpläne entgleisen lassen, Budgets erschöpfen und zu einem Produkt führen, das unfokussiert und übermäßig komplex ist. Die Ausdehnung kann aus sich entwickelnden Geschäftsbedürfnissen, Stakeholder-Anfragen, entdeckter Komplexität oder schlechter initialer Anforderungsdefinition resultieren. Anders als kontrollierte Scope-Änderungen geschieht Scope Creep graduell und oft ohne formale Anerkennung oder Anpassungen der Planung. Effektives Projektmanagement erfordert Wachsamkeit gegenüber Scope Creep und einen formalen Prozess zur Verwaltung vorgeschlagener Änderungen.

## Indicators ⟡

- Der Scope des Projekts dehnt sich konstant aus
- Ursprüngliche Projektanforderungen unterscheiden sich erheblich von den finalen Liefergegenständen
- Entwicklungsteams arbeiten an Features, die nicht in der ursprünglichen Spezifikation waren
- Projektzeitpläne dehnen sich weit über ursprüngliche Schätzungen hinaus, ohne formale Scope-Änderungsprozesse
- Das Team verpasst häufig Fristen
- Das Team wechselt konstant zwischen verschiedenen Aufgaben und Prioritäten
- Stakeholder fügen kontinuierlich „kleine" Anfragen hinzu, die sich zu größeren Änderungen anhäufen
- Feature-Sets wachsen organisch während der Entwicklung ohne Auswirkungsbewertung
- Es gibt viel Nacharbeit

## Symptoms ▲

- [Verzögerte Projektzeitpläne](verzoegerte-projektzeitplaene.md)
<br/>  Kontinuierlich wachsende Anforderungen schieben Projektliefertermine weiter hinaus, da mehr Arbeit hinzugefügt wird, ohne Zeitplananpassungen.
- [Budgetüberschreitungen](budgetueberschreitungen.md)
<br/>  Unkontrollierte Scope-Ausdehnung verbraucht mehr Ressourcen als ursprünglich geplant, was das Projekt sein Budget überschreiten lässt.
- [Feature-Aufblähung](feature-aufblaehung.md)
<br/>  Die kontinuierliche Hinzufügung ungeplanter Features resultiert in einem übermäßig komplexen Produkt, das das Kernwertversprechen verwässert.
- [Erhöhter Stress und Burnout](erhoehter-stress-und-burnout.md)
<br/>  Teams sehen sich zunehmendem Druck ausgesetzt, da wachsender Scope innerhalb ursprünglicher Zeitpläne und Ressourcen geliefert werden muss, was zu Überarbeitung führt.
- [Qualitätskompromisse](qualitaetskompromisse.md)
<br/>  Während sich der Scope ohne zusätzliche Zeit oder Ressourcen ausdehnt, werden Qualitätsstandards gesenkt, um das wachsende Feature-Set zu berücksichtigen.
- [Unvollständige Projekte](unvollstaendige-projekte.md)
<br/>  Projekte, die von Scope-Ausdehnung überwältigt werden, können aufgegeben oder unvollständig gelassen werden, wenn sie unhandhabbar werden.

## Causes ▼

- [Kein formaler Änderungskontrollprozess](kein-formaler-aenderungskontrollprozess.md)
<br/>  Ohne einen formalen Prozess zur Bewertung und Genehmigung von Änderungen werden neue Anfragen informell ohne Auswirkungsanalyse hinzugefügt.
- [Übermäßige Gefälligkeit gegenüber Stakeholdern](uebermaessige-gefaelligkeit-gegenueber-stakeholdern.md)
<br/>  Teams stimmen jeder Stakeholder-Anfrage zu, ohne zurückzudrängen oder Kompromisse zu erklären, was Anforderungen unkontrolliert wachsen lässt.
- [Unzureichende Anforderungserhebung](unzureichende-anforderungserhebung.md)
<br/>  Schlechte initiale Anforderungsdefinition führt zu kontinuierlicher Entdeckung fehlender Anforderungen während der Entwicklung, was Scope-Ausdehnung antreibt.
- [Schlechte Projektsteuerung](schlechte-projektsteuerung.md)
<br/>  Schwaches Projektmonitoring versäumt es, graduelle Scope-Ausdehnung zu erkennen, bis sie Zeitpläne und Budgets bereits erheblich beeinflusst hat.

## Detection Methods ○

- **Änderungsanfragen verfolgen:** Führen eines Protokolls aller neuen Feature-Anfragen und Änderungen an bestehenden Anforderungen
- **Scope-Änderungsverfolgung:** Überwachung von Ergänzungen und Modifikationen an ursprünglichen Projektanforderungen
- **Zeitplan-vs.-Scope-Analyse:** Vergleich des ursprünglichen Scopes und Zeitplans mit tatsächlichen Liefergegenständen und Dauer
- **Vergleich Plan vs. Ist:** Regelmäßiger Vergleich des Projektfortschritts mit dem ursprünglichen Plan, um zu sehen, wie sehr sich der Scope geändert hat
- **Velocity-Verfolgung:** In einem agilen Team kann ein Rückgang der Velocity ein Zeichen dafür sein, dass das Team mit ungeplanter Arbeit belastet wird (siehe [Langsame Entwicklungsgeschwindigkeit](langsame-entwicklungsgeschwindigkeit.md))
- **Feature-Anfragen-Analyse:** Verfolgung informeller Feature-Anfragen und ihrer Auswirkung auf den Projekt-Scope
- **Aufwandsabweichungsverfolgung:** Überwachung des tatsächlichen Aufwands im Vergleich zu ursprünglichen Schätzungen
- **Stakeholder-Anfragemuster:** Analyse der Häufigkeit und Art zusätzlicher Anfragen von Stakeholdern
- **Stakeholder-Feedback:** Wenn Stakeholder ständig fragen „Ist es schon fertig?", kann dies ein Zeichen dafür sein, dass ihre Erwartungen nicht mit der Realität des Projekts übereinstimmen

## Examples

Ein Team baut ein einfaches internes Dashboard für das Vertriebsteam. Anfangs besteht die einzige Anforderung darin, eine Liste von Kunden anzuzeigen. Dann fragt ein Stakeholder, ob auch der Gesamtumsatz für jeden Kunden angezeigt werden kann. Dann fragt ein anderer Stakeholder nach einem Umsatzdiagramm über die Zeit. Bald ist das einfache Dashboard zu einem komplexen Business-Intelligence-Werkzeug geworden, und das Projekt liegt Monate hinter dem Zeitplan. In einem anderen Fall befindet sich ein Projekt in seiner letzten Entwicklungswoche. Eine Führungskraft sieht eine Demo und sagt: „Das ist großartig, aber es wäre perfekt, wenn wir nur noch eine Sache hinzufügen könnten..." Das Team, das die Führungskraft zufriedenstellen möchte, stimmt der Änderung zu, was den Launch am Ende um einen Monat verzögert.

Ein Kundenportal-Projekt, das ursprünglich für einfache Kontoansicht und Passwort-Reset geplant war, wächst auf erweiterte Berichterstattung, Dokumenten-Upload, Zahlungsverarbeitung und mobile Optimierung, wenn Stakeholder frühe Prototypen sehen und zusätzliche Funktionalität anfragen. Der ursprüngliche 3-Monats-Zeitplan wird zu 8 Monaten, aber der Fristendruck bleibt bestehen, weil der Launch an eine Marketingkampagne gekoppelt war. Ein weiteres Beispiel betrifft ein internes Werkzeug-Projekt, bei dem die initiale Anforderung für einfache Dateneingabe auf Workflow-Management, Genehmigungsprozesse, Integration mit fünf externen Systemen und individuelle Berichterstattung erweitert wird, wenn verschiedene Abteilungen das Potenzial sehen und ihre eigenen Features einbezogen haben möchten. Dies ist ein sehr häufiges Problem in Softwareprojekten und einer der Hauptgründe, warum sie scheitern. Es ist besonders verbreitet in Organisationen mit schwacher Projektmanagement-Disziplin.
