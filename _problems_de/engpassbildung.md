---
title: Engpassbildung
description: Bestimmte Teammitglieder, Prozesse oder Systemkomponenten werden zu
  Einschränkungen, die den Gesamtfluss und die Produktivität der Entwicklungsarbeit
  begrenzen.
category:
- Performance
- Process
- Team
related_problems:
- slug: work-queue-buildup
  similarity: 0.8
- slug: maintenance-bottlenecks
  similarity: 0.75
- slug: work-blocking
  similarity: 0.75
- slug: review-bottlenecks
  similarity: 0.7
- slug: capacity-mismatch
  similarity: 0.65
- slug: process-design-flaws
  similarity: 0.65
solutions:
- event-driven-architecture
- parallelization
- pipelining
- read-replicas
- specialized-hardware
- streaming
- self-service-developer-platform
- value-stream-mapping
- work-in-progress-limits
- knowledge-rotation
- delivery-performance-metrics
layout: problem
lang: de
en_slug: bottleneck-formation
---

## Description

Engpassbildung entsteht, wenn bestimmte Einzelpersonen, Prozesse oder Systemkomponenten zu begrenzenden Faktoren werden, die den Gesamtdurchsatz und die Effizienz der Entwicklungsarbeit einschränken. Diese Engpässe erzeugen Warteschlangen, Verzögerungen und Abhängigkeiten, die den Fortschritt des gesamten Teams verlangsamen. Engpässe können sich um Personen mit Spezialwissen, Freigabeprozesse, gemeinsam genutzte Ressourcen oder technische Einschränkungen bilden.

## Indicators ⟡

- Arbeit staut sich durchgängig an, während auf bestimmte Personen oder Prozesse gewartet wird
- Die Teamgeschwindigkeit ist durch die Kapazität bestimmter Teammitglieder begrenzt
- Bestimmte Prozesse dauern unverhältnismäßig lange im Vergleich zu umgebenden Aktivitäten
- Der Arbeitsfluss ist unregelmäßig, mit Wartephasen gefolgt von Phasen der Eile
- Die Teamproduktivität variiert erheblich je nach Verfügbarkeit des Engpasses

## Symptoms ▲

- [Kaskadierende Verzögerungen](kaskadierende-verzoegerungen.md)
<br/>  Engpässe verzögern Liefergegenstände, von denen andere Teams abhängen, was Verzögerungen über Projekte hinweg fortpflanzt.
- [Aufstauung von Arbeitswarteschlangen](aufstauung-von-arbeitswarteschlangen.md)
<br/>  Arbeit häuft sich an, während auf die Engpassressource gewartet wird, was wachsende Warteschlangen erzeugt, die die Gesamtlieferung verzögern.
- [Verpasste Termine](verpasste-termine.md)
<br/>  Wenn der Durchsatz durch Engpässe eingeschränkt ist, verrutschen Projektzeitpläne, da Arbeit nicht im benötigten Tempo fortschreiten kann.
- [Entwicklerfrustration und Burnout](entwicklerfrustration-und-burnout.md)
<br/>  Teammitglieder werden frustriert, wenn sie untätig darauf warten, dass Engpassressourcen verfügbar werden.
- [Overhead durch Kontextwechsel](overhead-durch-kontextwechsel.md)
<br/>  Entwickler, die gezwungen sind, zwischen Aufgaben zu wechseln, während sie auf die Lösung des Engpasses warten, verlieren Produktivität durch Kontextwechsel.

## Causes ▼

- [Wissenssilos](wissenssilos.md)
<br/>  Wenn kritisches Wissen auf eine Person konzentriert ist, wird sie zum Engpass für alle Entscheidungen, die diese Expertise erfordern.
- [Single Points of Failure](single-points-of-failure.md)
<br/>  Nur eine Person oder einen Prozess zu haben, der kritische Funktionen ausführen kann, erzeugt inhärentes Engpassrisiko.
- [Freigabe-Abhängigkeiten](freigabe-abhaengigkeiten.md)
<br/>  Verpflichtende Freigabe-Workflows durch bestimmte Personen erzeugen Engpässe, wenn diese Personen nicht verfügbar sind.
- [Kapazitäts-Fehlanpassung](kapazitaets-fehlanpassung.md)
<br/>  Wenn die Kapazität an unterschiedlichen Prozessstufen nicht zur Nachfrage passt, werden eingeschränkte Stufen zu Engpässen.

## Detection Methods ○

- **Flussanalyse:** Nachverfolgung von Arbeitselementen durch den Entwicklungsprozess, um festzustellen, wo Verzögerungen auftreten
- **Kapazitätsauslastungs-Monitoring:** Messung der Auslastungsraten über verschiedene Teammitglieder und Prozesse hinweg
- **Warteschlangenlängen-Tracking:** Beobachtung, wie sich Arbeit in verschiedenen Stufen der Entwicklungspipeline anhäuft
- **Durchlaufzeit-Messung:** Analyse, wie lange Arbeitselemente bis zur Fertigstellung brauchen und wo Zeit verbracht wird
- **Abhängigkeits-Mapping:** Identifikation kritischer Abhängigkeiten, die Einschränkungen des Arbeitsflusses erzeugen

## Examples

Der Fortschritt eines Entwicklungsteams wird durchgängig durch seine Senior-Architektin begrenzt, die alle bedeutenden Design-Entscheidungen überprüfen und genehmigen muss. Arbeit staut sich an, während auf ihre Verfügbarkeit gewartet wird, und Teammitglieder warten oft Tage auf Design-Anleitung, bevor sie mit der Umsetzung fortfahren können. Trotz sechs Entwicklern ist der effektive Durchsatz des Teams durch die Kapazität einer Person für Design-Reviews und architektonische Entscheidungen eingeschränkt. Ein weiteres Beispiel betrifft einen Deployment-Prozess, der die manuelle Genehmigung des Operations-Teams erfordert und nur während bestimmter Wartungsfenster durchgeführt werden kann. Entwicklungsarbeit wird schnell abgeschlossen, aber Features warten auf Deployment-Slots, was einen erheblichen Engpass zwischen Entwicklungsabschluss und Wertlieferung erzeugt. Das Team erkennt, dass sein Deployment-Engpass seine Fähigkeit einschränkt, Kunden effizient Wert zu liefern.
