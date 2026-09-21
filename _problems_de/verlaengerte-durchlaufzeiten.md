---
title: Verlängerte Durchlaufzeiten
description: Die Zeit vom Beginn der Arbeit bis zu ihrer Fertigstellung und Auslieferung
  wird deutlich länger als die tatsächlich benötigte Arbeitszeit.
category:
- Process
related_problems:
- slug: extended-review-cycles
  similarity: 0.7
- slug: delayed-project-timelines
  similarity: 0.7
- slug: work-queue-buildup
  similarity: 0.7
- slug: extended-research-time
  similarity: 0.7
- slug: long-build-and-test-times
  similarity: 0.7
- slug: long-release-cycles
  similarity: 0.65
solutions:
- ci-cd-pipeline
- small-change-batches
- work-in-progress-limits
- trunk-based-development
- continuous-delivery
- capacity-based-planning
- code-review-guidelines
- value-stream-mapping
- delivery-performance-metrics
- fast-feedback-loops
- self-service-developer-platform
layout: problem
lang: de
en_slug: extended-cycle-times
---

## Description

Verlängerte Durchlaufzeiten entstehen, wenn die Gesamtzeit von der Aufgabeninitiierung bis zur Fertigstellung erheblich länger ist als die tatsächlich für die Aufgabe aufgewendete Zeit. Dies deutet darauf hin, dass Arbeit mehr Zeit in Warteschlangen verbringt, durch Abhängigkeiten blockiert oder in Prozessen festgehalten wird, als aktiv bearbeitet zu werden. Verlängerte Durchlaufzeiten verringern die Reaktionsfähigkeit auf Geschäftsbedürfnisse und deuten auf Ineffizienzen im Entwicklungsprozess hin.

## Indicators ⟡

- Die Gesamtzeit vom Aufgabenbeginn bis zur Fertigstellung ist um ein Vielfaches länger als die tatsächliche Arbeitszeit
- Arbeitselemente verbringen mehr Zeit "in Bearbeitung" als aktiv bearbeitet zu werden
- Aufgaben verbleiben über längere Zeit im selben Status, ohne Fortschritt
- Kleine Änderungen brauchen Wochen oder Monate zur Fertigstellung, obwohl sie nur Stunden Arbeit erfordern
- Das Team kann erhebliche Wartezeiten in seinem Arbeitsprozess identifizieren

## Symptoms ▲

- [Verzögerte Wertlieferung](verzoegerte-wertlieferung.md)
<br/>  Wenn Durchlaufzeiten verlängert sind, warten Nutzer viel länger, um Features und Fixes zu erhalten, was die Reaktionsfähigkeit des Produkts auf Bedürfnisse verringert.
- [Erhöhte Time-to-Market](erhoehte-time-to-market.md)
<br/>  Verlängerte Durchlaufzeiten übersetzen sich direkt in längere Time-to-Market für neue Features und Produkte.
- [Frustration der Stakeholder](frustration-der-stakeholder.md)
<br/>  Stakeholder werden frustriert, wenn kleine Änderungen Wochen oder Monate brauchen, um die Produktion zu erreichen, obwohl nur Stunden Arbeit nötig wären.
- [Verpasste Termine](verpasste-termine.md)
<br/>  Verlängerte Wartezeiten während des gesamten Prozesses führen dazu, dass Aufgaben durchgängig ihre geschätzten Fertigstellungstermine verpassen.
- [Langsame Entwicklungsgeschwindigkeit](langsame-entwicklungsgeschwindigkeit.md)
<br/>  Lange Durchlaufzeiten verringern die scheinbare Geschwindigkeit des Teams, da Arbeitselemente in Warteschlangen sitzen, statt abgeschlossen zu werden.
- [Wettbewerbsnachteil](wettbewerbsnachteil.md)
<br/>  Langsame Lieferzyklen verhindern, dass das Team schnell auf Marktveränderungen und Nutzerbedürfnisse reagiert, was Wettbewerbern einen Vorteil verschafft.

## Causes ▼

- [Verlängerte Review-Zyklen](verlaengerte-review-zyklen.md)
<br/>  Mehrere Runden von Code-Review-Feedback und Überarbeitung fügen dem Gesamtzyklus erhebliche Wartezeit hinzu.
- [Freigabe-Abhängigkeiten](freigabe-abhaengigkeiten.md)
<br/>  Erforderliche Freigaben durch bestimmte Personen schaffen Warteschlangen und Verzögerungen, die die Gesamtdurchlaufzeit verlängern.
- [Review-Engpässe](review-engpaesse.md)
<br/>  Der Code-Review-Prozess, der zum Engpass wird, fügt Wartezeit hinzu, die die Gesamtdurchlaufzeiten aufbläht.
- [Lange Release-Zyklen](lange-release-zyklen.md)
<br/>  Seltene Release-Fenster bedeuten, dass abgeschlossene Arbeit auf die nächste Deployment-Gelegenheit wartet.
- [Komplexer Deployment-Prozess](komplexer-deployment-prozess.md)
<br/>  Manuelle und komplizierte Deployment-Prozesse fügen erhebliche Zeit zwischen Code-Fertigstellung und Produktionslieferung hinzu.

## Detection Methods ○

- **Durchlaufzeit-Messung:** Nachverfolgung der Gesamtzeit vom Arbeitsbeginn bis zur Fertigstellung
- **Flusseffizienzanalyse:** Berechnung des Verhältnisses von Arbeitszeit zu Gesamtdurchlaufzeit
- **Wartezeit-Tracking:** Identifikation, wie viel Zeit Aufgaben im Warten vs. in Bearbeitung verbringen
- **Prozessschritt-Analyse:** Messung der Zeit, die an jeder Stufe des Entwicklungsprozesses verbracht wird
- **Vergleichsanalyse:** Vergleich von Durchlaufzeiten für ähnliche Arbeitselemente zur Identifikation von Mustern

## Examples

Eine einfache Fehlerbehebung, die 2 Stunden Entwicklungszeit erfordert, braucht 6 Wochen, um die Produktion zu erreichen, aufgrund verlängerter Code-Review-Warteschlangen, Deployment-Genehmigungsprozesse und monatlicher Release-Zyklen. Die tatsächliche Arbeit wird schnell abgeschlossen, aber der Fix verbringt die meiste Zeit in verschiedenen Warteschlangen und Genehmigungsprozessen. Ein weiteres Beispiel betrifft eine Feature-Anfrage, die von der Genehmigung bis zur Lieferung 3 Monate braucht, obwohl nur 1 Woche tatsächlicher Entwicklungsarbeit nötig ist. Die verlängerte Durchlaufzeit wird durch das Warten auf Design-Genehmigung, Entwicklungswarteschlangen-Rückstände, Test-Engpässe und Einschränkungen bei der Deployment-Planung verursacht.
