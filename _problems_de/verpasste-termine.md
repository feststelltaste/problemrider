---
title: Verpasste Termine
description: Projekte überschreiten regelmäßig ihre geschätzten Fertigstellungszeiten,
  und Teams verfehlen konsequent Sprint-Ziele und Lieferzusagen.
category:
- Business
- Process
- Team
related_problems:
- slug: delayed-project-timelines
  similarity: 0.85
- slug: constantly-shifting-deadlines
  similarity: 0.75
- slug: unrealistic-deadlines
  similarity: 0.75
- slug: slow-development-velocity
  similarity: 0.7
- slug: poor-planning
  similarity: 0.7
- slug: cascade-delays
  similarity: 0.7
solutions:
- evolutionary-requirements-development
- iterative-development
- short-iteration-cycles
- capacity-based-planning
- regular-stakeholder-demonstrations
- explicit-prioritization-framework
- work-in-progress-limits
- definition-of-ready
- value-stream-mapping
layout: problem
lang: de
en_slug: missed-deadlines
---

## Description

Verpasste Termine treten auf, wenn Entwicklungsteams konsequent Arbeit nicht innerhalb vereinbarter Zeitrahmen liefern, seien es Sprint-Ziele, Release-Termine oder Projektmeilensteine. Dieses Muster deutet auf zugrunde liegende Probleme bei Schätzung, Planung, Ausführung oder externen Faktoren hin, die Teams daran hindern, ihre Zusagen einzuhalten. Chronisch verpasste Termine untergraben das Stakeholder-Vertrauen und können einen Kreislauf aus erhöhtem Druck und weiteren Verzögerungen schaffen.

## Indicators ⟡

- Sprint-Ziele werden über mehrere Iterationen hinweg konsequent nicht erreicht
- Projekt-Liefertermine werden regelmäßig nach hinten verschoben
- Die Team-Geschwindigkeit ist konsequent niedriger als die geplante Kapazität
- Stakeholder äußern Frustration über unvorhersehbare Lieferzeitpläne
- Teams verbringen erhebliche Zeit damit, zu erklären, warum Arbeit länger dauerte als erwartet

## Symptoms ▲

- [Unzufriedenheit der Stakeholder](unzufriedenheit-der-stakeholder.md)
<br/>  Wiederholt verpasste Termine untergraben das Stakeholder-Vertrauen und schaffen Reibung zwischen Geschäft und Entwicklungsteams.
- [Ständig verschobene Termine](staendig-verschobene-termine.md)
<br/>  Wenn Termine regelmäßig verpasst werden, beginnen Teams, Zeitpläne präventiv anzupassen, was ein Muster verschobener Zusagen schafft.
- [Verzögerte Wertlieferung](verzoegerte-wertlieferung.md)
<br/>  Verpasste Termine verzögern direkt die Lieferung von Geschäftswert an Nutzer und die Organisation.
- [Erhöhter Stress und Burnout](erhoehter-stress-und-burnout.md)
<br/>  Der Druck und die Frustration durch wiederholt verpasste Zusagen tragen zu Team-Stress und Burnout bei.
- [Termindruck](termindruck.md)
<br/>  Nach verpassten Terminen sehen sich Teams verstärktem Druck ausgesetzt, bei nachfolgenden Zusagen zu liefern, was einen Teufelskreis schafft.

## Causes ▼

- [Unrealistische Termine](unrealistische-termine.md)
<br/>  Zusagen, die ohne realistische Einschätzung von Aufwand und Komplexität gemacht werden, stellen Teams so auf, dass sie Termine verpassen.
- [Schlechte Planung](schlechte-planung.md)
<br/>  Unzureichende Projektplanung, die Risiken, Abhängigkeiten und Overhead nicht berücksichtigt, führt zu unterschätzten Zeitplänen.
- [Scope Creep](scope-creep.md)
<br/>  Unkontrollierte Hinzufügung von Anforderungen während der Entwicklung erhöht die Arbeit über das ursprünglich Geschätzte hinaus.
- [Anforderungsmehrdeutigkeit](anforderungsmehrdeutigkeit.md)
<br/>  Mehrdeutige Anforderungen führen zu Nacharbeit und der Entdeckung unausgesprochener Bedürfnisse, was Zeit verbraucht, die nicht in Schätzungen berücksichtigt wurde.
- [Langsame Entwicklungsgeschwindigkeit](langsame-entwicklungsgeschwindigkeit.md)
<br/>  Wenn die Entwicklungsgeschwindigkeit aufgrund technischer Schulden oder Prozessprobleme niedriger als erwartet ist, werden Termine verpasst.
- [Langsame Feature-Entwicklung](langsame-feature-entwicklung.md)
<br/>  Wenn die Feature-Implementierung konsequent länger dauert als geschätzt, verpassen Teams die Liefertermine, die auf Grundlage dieser Schätzungen festgelegt wurden.
- [Analyse-Lähmung](analyse-laehmung.md)
<br/>  Teams, die stecken bleiben beim Analysieren von Optionen statt sie zu implementieren, verbrauchen die für die Lieferung budgetierte Zeit, was zu verpassten Terminen führt.

## Detection Methods ○

- **Nachverfolgung der Sprint-Abschlussrate:** Überwachung des Prozentsatzes erreichter Sprint-Ziele über die Zeit
- **Analyse der Liefertermin-Abweichung:** Nachverfolgung tatsächlicher vs. geplanter Liefertermine für Projekte
- **Geschwindigkeitstrend-Analyse:** Vergleich geplanter vs. tatsächlicher Team-Geschwindigkeit über Sprints hinweg
- **Stakeholder-Zufriedenheitsbefragungen:** Bewertung des Vertrauens von Geschäftspartnern in Lieferzeitpläne
- **Ursachenanalyse:** Systematische Analyse der Gründe für spezifische verpasste Termine

## Examples

Ein mobiles App-Entwicklungsteam schließt konsequent nur 60 % seiner geplanten Sprint-Arbeit ab, was dazu führt, dass Feature-Releases im Durchschnitt um 2-3 Sprints verzögert werden. Untersuchung zeigt, dass ihre Schätzungen die Komplexität des Testens über mehrere Gerätetypen und Betriebssystemversionen hinweg nicht berücksichtigen, und sie werden häufig durch Produktionssupport-Probleme unterbrochen, die nicht in der Sprint-Planung berücksichtigt wurden. Ein weiteres Beispiel betrifft ein Webentwicklungsteam, das konsequent Projekttermine verpasst, weil ihre Schätzungen ideale Entwicklungsbedingungen voraussetzen, sie aber 40 % ihrer Zeit mit Infrastrukturproblemen, unklaren Anforderungen und Abhängigkeitsverzögerungen verbringen, die in der Projektplanung nicht berücksichtigt wurden.
