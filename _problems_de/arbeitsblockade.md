---
title: Arbeitsblockade
description: Entwicklungsaufgaben können ohne ausstehende Genehmigungen nicht fortschreiten,
  was Engpässe und Verzögerungen im Entwicklungsprozess schafft.
category:
- Management
- Process
related_problems:
- slug: approval-dependencies
  similarity: 0.85
- slug: bottleneck-formation
  similarity: 0.75
- slug: delayed-decision-making
  similarity: 0.75
- slug: decision-avoidance
  similarity: 0.7
- slug: work-queue-buildup
  similarity: 0.7
- slug: inefficient-processes
  similarity: 0.7
solutions:
- sustainable-pace-practices
- team-autonomy-and-empowerment
- decision-rights-and-escalation
- explicit-prioritization-framework
- team-boundaries-aligned-to-architecture
- work-in-progress-limits
- definition-of-ready
- value-stream-mapping
- self-service-developer-platform
layout: problem
lang: de
en_slug: work-blocking
---

## Description

Arbeitsblockade tritt auf, wenn Entwicklungsaufgaben nicht voranschreiten können, weil sie Genehmigungen, Entscheidungen oder Inputs erfordern, die verzögert oder nicht verfügbar sind. Dies schafft einen Engpasseffekt, bei dem Entwickler und Teams untätig sitzen oder zu weniger produktiver Arbeit wechseln, während sie auf Erlaubnis warten, fortzufahren. Arbeitsblockade deutet oft auf übermäßig zentralisierte Entscheidungsfindung, unklare Autoritätsstrukturen oder Prozesse hin, die Kontrolle über Produktivität priorisieren.

## Indicators ⟡

- Entwickler berichten während Stand-up-Meetings häufig, „blockiert" zu sein
- Aufgaben bleiben für längere Zeit im Status „wartet auf Genehmigung"
- Die Team-Velocity sinkt aufgrund von Kontextwechsel während des Wartens auf Entscheidungen
- Entwickler arbeiten an niedrigprioritären Aufgaben, während höherprioritäre Arbeit blockiert ist
- Mehrere Teammitglieder sind von derselben Person oder demselben Prozess für Genehmigungen abhängig

## Symptoms ▲

- [Overhead durch Kontextwechsel](overhead-durch-kontextwechsel.md)
<br/>  Wenn Entwickler blockiert sind und auf Genehmigungen warten, wechseln sie zu niedrigprioritären Aufgaben, was kognitiven Overhead durch häufige Kontextwechsel verursacht.
- [Verzögerte Projektzeitpläne](verzoegerte-projektzeitplaene.md)
<br/>  Aufgaben, die im blockierten Status feststecken, verzögern direkt Projektmeilensteine und Lieferzeitpläne.
- [Aufstauung von Arbeitswarteschlangen](aufstauung-von-arbeitswarteschlangen.md)
<br/>  Blockierte Arbeitselemente häufen sich in Warteschlangen an, was Rückstände an Genehmigungs- und Entscheidungspunkten schafft.
- [Entwicklerfrustration und Burnout](entwicklerfrustration-und-burnout.md)
<br/>  Entwickler, die untätig auf Genehmigungen für wichtige Arbeit warten, erleben über die Zeit Frustration und sinkende Moral.
- [Verringerte Teamproduktivität](verringerte-teamproduktivitaet.md)
<br/>  Zeit, die mit dem Warten auf Genehmigungen statt der Produktion von Code verbracht wird, verringert direkt den Team-Output.
- [Workaround-Kultur](workaround-kultur.md)
<br/>  Wenn ordentliche Änderungen durch Genehmigungsprozesse blockiert werden, greifen Entwickler zu Workarounds, die den blockierenden Prozess umgehen.
- [Verschwendeter Entwicklungsaufwand](verschwendeter-entwicklungsaufwand.md)
<br/>  Wenn Entwickler blockiert sind, wechseln sie zu niedrigprioritären Aufgaben oder machen Arbeit, die invalidiert werden könnte, sobald die Blockade aufgehoben wird.

## Causes ▼

- [Freigabe-Abhängigkeiten](freigabe-abhaengigkeiten.md)
<br/>  Arbeitsblockade wird direkt durch Prozesse verursacht, die verlangen, dass bestimmte Personen genehmigen, bevor Arbeit fortschreiten kann.
- [Engpassbildung](engpassbildung.md)
<br/>  Zentralisierte Entscheidungsfindung oder knappe Reviewer-Verfügbarkeit schafft Engpässe, die Arbeitselemente blockieren.
- [Entscheidungsvermeidung](entscheidungsvermeidung.md)
<br/>  Wenn Entscheidungsträger Entscheidungen vermeiden oder aufschieben, bleibt Arbeit, die von diesen Entscheidungen abhängt, blockiert.
- [Mikromanagement-Kultur](mikromanagement-kultur.md)
<br/>  Exzessive Managementaufsicht, die Genehmigung für Routineentscheidungen verlangt, schafft unnötige Blockierung von Entwicklungsaufgaben.

## Detection Methods ○

- **Verfolgung der Blockierzeit:** Überwachung, wie viel Zeit Aufgaben im blockierten Status verbringen
- **Genehmigungswarteschlangen-Analyse:** Verfolgung von Volumen und Bearbeitungszeit verschiedener Arten von Genehmigungsanfragen
- **Entwicklerbefragungen:** Befragung von Teammitgliedern zu ihrer Erfahrung mit Genehmigungen und Entscheidungsautonomie
- **Stand-up-Meeting-Analyse:** Zählung der Häufigkeit von „blockiert"-Statusberichten und deren Gründe
- **Entscheidungsautoritäts-Mapping:** Identifikation von Entscheidungsarten, die Genehmigung erfordern, versus solchen, die unabhängig getroffen werden können
- **Flusseffizienzmessung:** Berechnung des Prozentsatzes der Zeit, in der Arbeitselemente aktiv fortschreiten, versus warten

## Examples

Ein Entwicklungsteam benötigt Genehmigung vom Architekturkomitee für jegliche Datenbankschemaänderungen. Das Komitee trifft sich einmal pro Woche, und Entscheidungen erfordern oft zusätzliche Dokumentation oder Klärung, was zu mehrwöchigen Verzögerungen für einfache Änderungen wie das Hinzufügen eines Index führt. Entwickler enden damit, an weniger wichtigen Aufgaben zu arbeiten, während kritische Performance-Verbesserungen blockiert sind. Ein weiteres Beispiel betrifft ein Mobile-App-Team, das UI-Design-Genehmigung von einem Design-Direktor erhalten muss, der häufig reist. Einfache Layout-Anpassungen, die in Stunden implementiert werden könnten, warten stattdessen Wochen auf Genehmigung, was Entwickler zwingt, unvollständige Designs zu umgehen oder Feature-Releases zu verzögern.
