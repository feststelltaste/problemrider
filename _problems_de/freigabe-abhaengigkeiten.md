---
title: Freigabe-Abhängigkeiten
description: Der Arbeitsfortschritt wird häufig durch die Notwendigkeit von Freigaben
  bestimmter Personen blockiert, was Engpässe und Verzögerungen erzeugt.
category:
- Dependencies
- Process
- Process
related_problems:
- slug: work-blocking
  similarity: 0.85
- slug: delayed-decision-making
  similarity: 0.7
- slug: knowledge-dependency
  similarity: 0.7
- slug: cascade-delays
  similarity: 0.7
- slug: rushed-approvals
  similarity: 0.7
- slug: decision-avoidance
  similarity: 0.65
solutions:
- clear-ownership-model
- formal-change-control-process
- iterative-development
- product-owner
- team-boundaries-aligned-to-architecture
- value-stream-mapping
- self-service-developer-platform
- decision-rights-and-escalation
- delivery-performance-metrics
- change-impact-analysis
- executive-sponsorship
layout: problem
lang: de
en_slug: approval-dependencies
---

## Description

Freigabe-Abhängigkeiten entstehen, wenn Arbeit nicht ohne ausdrückliche Genehmigung bestimmter Personen fortschreiten kann, was Engpässe und Verzögerungen im Entwicklungsprozess erzeugt. Dieses Problem ist besonders akut, wenn Freigaben für Routineentscheidungen erforderlich sind, wenn Genehmigungsinstanzen häufig nicht verfügbar sind oder wenn Freigabeprozesse unnötig komplex sind. Diese Abhängigkeiten können ganze Teams zum Stillstand bringen, während sie auf Autorisierung warten.

## Indicators ⟡

- Die Arbeit stoppt häufig, während auf Freigaben bestimmter Personen gewartet wird
- Freigabeanfragen häufen sich schneller an, als sie bearbeitet werden können
- Einfache Entscheidungen erfordern die Genehmigung des oberen Managements
- Teammitglieder verbringen erhebliche Zeit damit, Freigaben einzuholen, statt produktiv zu arbeiten
- Freigabeprozesse variieren inkonsistent, je nachdem, wer verfügbar ist

## Symptoms ▲

- [Verzögerte Projektzeitpläne](verzoegerte-projektzeitplaene.md)
<br/>  Durch ausstehende Freigaben blockierte Arbeit verzögert Projektzeitpläne direkt, da Aufgaben nicht fortschreiten können.
- [Kaskadierende Verzögerungen](kaskadierende-verzoegerungen.md)
<br/>  Eine einzelne blockierte Freigabe kann nachgelagerte Aufgaben verzögern und so kaskadierende Verzögerungen im gesamten Projekt erzeugen.
- [Verringerte Teamproduktivität](verringerte-teamproduktivitaet.md)
<br/>  Teammitglieder sind untätig, während sie auf Freigaben warten, was die Gesamtproduktivität des Teams direkt verringert.
- [Entwicklerfrustration und Burnout](entwicklerfrustration-und-burnout.md)
<br/>  Wiederholt durch Freigabeprozesse blockiert zu werden, frustriert Entwickler und trägt zu Burnout bei.
- [Overhead durch Kontextwechsel](overhead-durch-kontextwechsel.md)
<br/>  Entwickler, die gezwungen sind, während des Wartens auf Freigaben zu anderen Aufgaben zu wechseln, verlieren Fokus und Effizienz.
- [Übereilte Genehmigungen](uebereilte-genehmigungen.md)
<br/>  Der Rückstau angehäufter Freigabeanfragen bringt Genehmiger dazu, Entscheidungen zu überstürzen.
- [Engpassbildung](engpassbildung.md)
<br/>  Die Konzentration der Freigabebefugnis auf wenige Personen erzeugt strukturelle Engpässe, die Arbeit blockieren.

## Causes ▼

- [Mikromanagement-Kultur](mikromanagement-kultur.md)
<br/>  Eine Kultur des Mikromanagements verlangt Freigaben für Routineentscheidungen, die Teams eigenständig treffen können sollten.
- [Angst vor Scheitern](angst-vor-scheitern.md)
<br/>  Organisationen, die Angst vor Fehlern haben, schaffen übermäßige Freigabeanforderungen als Risikominderungsstrategie.

## Detection Methods ○

- **Freigabe-Warteschlangen-Tracking:** Beobachtung, wie viele Freigabeanfragen ausstehend sind und wie lange
- **Analyse von Arbeitsblockaden:** Nachverfolgung, wie oft Arbeit durch das Warten auf Freigaben blockiert wird
- **Freigabe-Reaktionszeit:** Messung, wie lange es dauert, Freigaben für verschiedene Arten von Entscheidungen zu erhalten
- **Entscheidungstyp-Analyse:** Kategorisierung, welche Arten von Entscheidungen eine Freigabe erfordern und welche nicht
- **Auswirkung auf die Teamproduktivität:** Bewertung, wie Freigabe-Abhängigkeiten die Gesamtproduktivität des Teams beeinflussen

## Examples

Ein Entwicklungsteam muss für jede Datenbankschemaänderung die Genehmigung seines Direktors einholen, selbst für kleinere wie das Hinzufügen eines Index oder das Umbenennen einer Spalte. Der Direktor ist häufig in Meetings oder auf Reisen, sodass Anfragen für Schemaänderungen oft 1-2 Wochen auf Genehmigung warten, während die Entwicklungsarbeit blockiert ist. Einfache Performance-Optimierungen, die in einer Stunde umgesetzt werden könnten, dauern stattdessen aufgrund des Freigabe-Engpasses Wochen. Ein weiteres Beispiel betrifft ein Team, bei dem jedes Produktions-Deployment sowohl die Genehmigung des Security-Teams als auch des Operations-Teams erfordert, aber es keine Koordination zwischen diesen Genehmigungen gibt, sodass Deployments oft von einem Team genehmigt, aber vom anderen verzögert werden, was zu unvorhersehbaren Deployment-Zeitplänen führt und Entwickler zwingt, mehrere Versionen ihrer Änderungen zu pflegen.
