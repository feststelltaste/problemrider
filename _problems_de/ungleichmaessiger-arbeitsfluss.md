---
title: Ungleichmäßiger Arbeitsfluss
description: Arbeit schreitet in unregelmäßigen Schüben voran statt gleichmäßig durch
  den Entwicklungsprozess zu fließen.
category:
- Process
- Team
related_problems:
- slug: uneven-workload-distribution
  similarity: 0.7
- slug: capacity-mismatch
  similarity: 0.65
- slug: work-queue-buildup
  similarity: 0.65
- slug: work-blocking
  similarity: 0.6
- slug: bottleneck-formation
  similarity: 0.6
- slug: extended-cycle-times
  similarity: 0.6
solutions:
- sustainable-pace-practices
- work-in-progress-limits
- short-iteration-cycles
- capacity-based-planning
- explicit-prioritization-framework
- continuous-delivery
- value-stream-mapping
- delivery-performance-metrics
- team-retrospectives
- clear-roles-and-ownership
layout: problem
lang: de
en_slug: uneven-work-flow
---

## Description

Ungleichmäßiger Arbeitsfluss tritt auf, wenn Entwicklungsarbeit unregelmäßig voranschreitet, mit Perioden intensiver Aktivität gefolgt von Perioden des Wartens oder langsamen Fortschritts. Dieses Muster deutet auf Probleme mit Prozessdesign, Ressourcenzuweisung oder Abhängigkeitsmanagement hin, die gleichmäßigen, vorhersehbaren Arbeitsfortschritt verhindern. Ungleichmäßiger Fluss verringert die Gesamteffizienz und macht Planung und Vorhersage schwierig.

## Indicators ⟡

- Die Arbeitsfertigstellung variiert dramatisch zwischen Zeiträumen
- Das Team wechselt zwischen sehr beschäftigt sein und wenig zu tun haben
- Manche Teammitglieder sind konsequent überlastet, während andere freie Kapazität haben
- Die Projektgeschwindigkeit variiert erheblich zwischen Sprints oder Zeiträumen
- Arbeit bleibt oft an bestimmten Stufen für unvorhersehbare Zeiträume stecken

## Symptoms ▲

- [Verpasste Termine](verpasste-termine.md)
<br/>  Unregelmäßiger Arbeitsfluss macht es unmöglich, Fertigstellungszeiten vorherzusagen, was zu verpassten Terminen führt, wenn Arbeit ins Stocken gerät.
- [Erhöhter Stress und Burnout](erhoehter-stress-und-burnout.md)
<br/>  Der Wechsel zwischen Leerlaufperioden und intensivem Ansturm schafft Stress und nicht nachhaltige Arbeitsmuster.
- [Qualitätskompromisse](qualitaetskompromisse.md)
<br/>  Wenn Arbeit in Aktivitätsschübe komprimiert wird, leidet die Qualität, weil während Spitzenzeiten keine Zeit für gründliche Arbeit bleibt.
- [Verringerte Teamproduktivität](verringerte-teamproduktivitaet.md)
<br/>  Das Stop-and-Go-Muster ungleichmäßigen Flusses verschwendet Kapazität während Leerlaufperioden und überlastet während Schüben, was den Gesamtoutput verringert.
- [Aufstauung von Arbeitswarteschlangen](aufstauung-von-arbeitswarteschlangen.md)
<br/>  Arbeit häuft sich an Engpassstufen während langsamer Perioden an, was Warteschlangen schafft, die zum unregelmäßigen Flussmuster beitragen.

## Causes ▼

- [Engpassbildung](engpassbildung.md)
<br/>  Engpässe an bestimmten Prozessstufen verursachen, dass Arbeit ins Stocken gerät, was das unregelmäßige Flussmuster schafft.
- [Freigabe-Abhängigkeiten](freigabe-abhaengigkeiten.md)
<br/>  Das Warten auf Freigaben von bestimmten Personen schafft unvorhersehbare Verzögerungen, die gleichmäßigen Arbeitsfluss stören.
- [Fehler im Prozessdesign](fehler-im-prozessdesign.md)
<br/>  Schlecht designte Prozesse mit fehlangepassten Kapazitäten an verschiedenen Stufen produzieren inhärent ungleichmäßigen Fluss.
- [Kapazitäts-Fehlanpassung](kapazitaets-fehlanpassung.md)
<br/>  Wenn die Kapazität an verschiedenen Stufen nicht der Nachfrage entspricht, wechselt Arbeit zwischen Aufstauung und freiem Fluss.

## Detection Methods ○

- **Flussvariabilitätsanalyse:** Messung der Variation in Arbeitsfertigstellungsraten über die Zeit
- **Zykluszeitverteilung:** Analyse der Verteilung von Zykluszeiten zur Identifikation unregelmäßiger Muster
- **Ressourcennutzungsverfolgung:** Überwachung, wie sich Ressourcennutzung über die Zeit ändert
- **Work-in-Progress-Monitoring:** Verfolgung, wie sich WIP-Niveaus ändern, und Identifikation von Anhäufungspunkten
- **Team-Velocity-Varianz:** Messung, wie stark die Team-Velocity zwischen Zeiträumen variiert

## Examples

Ein Softwareentwicklungsteam erlebt ein Muster, bei dem es den Großteil seiner Sprint-Arbeit in den letzten zwei Tagen abschließt, nachdem es die erste anderthalb Wochen damit verbracht hat, Blocker zu bewältigen, auf Freigaben zu warten und zwischen teilweise abgeschlossenen Aufgaben den Kontext zu wechseln. Dies schafft intensiven Druck am Sprint-Ende und macht es unmöglich, konsistente Qualität aufrechtzuerhalten oder Fertigstellungszeiten vorherzusagen. Ein weiteres Beispiel betrifft ein Datenverarbeitungsteam, bei dem Arbeit gleichmäßig fließt, bis sie die Datenvalidierungsstufe erreicht, wo ein einziger Experte alle Ausgaben überprüfen muss. Arbeit staut sich an dieser Stufe für Tage auf und wird dann in Eile verarbeitet, wenn der Experte verfügbar ist, was ein unregelmäßiges Stop-and-Go-Muster schafft, das den gesamten Prozess unvorhersehbar und stressig macht.
