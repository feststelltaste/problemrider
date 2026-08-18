---
title: Lange Release-Zyklen
description: Releases verzögern sich aufgrund langwieriger manueller Testphasen oder
  Fehlerentdeckungen in letzter Minute.
category:
- Management
- Process
- Testing
related_problems:
- slug: large-risky-releases
  similarity: 0.75
- slug: extended-review-cycles
  similarity: 0.7
- slug: increased-manual-testing-effort
  similarity: 0.7
- slug: delayed-bug-fixes
  similarity: 0.7
- slug: long-build-and-test-times
  similarity: 0.7
- slug: extended-cycle-times
  similarity: 0.65
solutions:
- ci-cd-pipeline
- continuous-delivery
- continuous-integration-and-delivery
- feature-toggles
- continuous-deployment
- value-stream-mapping
- delivery-performance-metrics
- fast-feedback-loops
- variant-consolidation
- explicit-extension-points
layout: problem
lang: de
en_slug: long-release-cycles
---

## Description

Lange Release-Zyklen treten auf, wenn die Zeit zwischen Software-Releases aufgrund langwieriger Testphasen, umfangreicher manueller Verifikationsprozesse oder häufiger später Entdeckung von Problemen im Release-Prozess exzessiv wird. Dieses Problem schafft einen Engpass bei der Wertlieferung an Nutzer und führt oft zu größeren, riskanteren Releases, die noch schwieriger zu testen und zu deployen sind. Lange Zyklen können sich selbst verstärken, während Teams versuchen, mehr Features in seltene Releases zu packen, was jedes Release noch größer und komplexer macht.

## Indicators ⟡
- Releases erfolgen monatlich, vierteljährlich oder noch seltener, obwohl sie regelmäßiger sein sollten
- Erhebliche Teile des Release-Zyklus werden für manuelles Testen oder Fehlerbehebung aufgewendet
- Release-Termine werden häufig wegen Qualitätsbedenken verschoben
- Große Mengen an Codeänderungen häufen sich zwischen Releases an
- Das Team verbringt Wochen mit der Vorbereitung jedes Releases

## Symptoms ▲

- [Große, riskante Releases](grosse-riskante-releases.md)
<br/>  Lange Zyklen führen dazu, dass sich Änderungen anhäufen, was zu größeren Releases führt, die mehr Risiko tragen und schwerer zu testen sind.
- [Verzögerte Wertlieferung](verzoegerte-wertlieferung.md)
<br/>  Nutzer und Kunden warten Monate auf Features und Fixes, die bereits fertig, aber in unveröffentlichtem Code gefangen sind.
- [Erhöhte Time-to-Market](erhoehte-time-to-market.md)
<br/>  Lange Release-Zyklen verlängern direkt die Zeit von der Feature-Fertigstellung bis zur Nutzerverfügbarkeit, was die Wettbewerbsposition schädigt.
- [Kundenunzufriedenheit](kundenunzufriedenheit.md)
<br/>  Nutzer werden frustriert, während sie durch lange Release-Zyklen auf angeforderte Features und Fehlerbehebungen warten.
- [Wettbewerbsnachteil](wettbewerbsnachteil.md)
<br/>  Wettbewerber mit schnelleren Release-Kadenzen können schneller auf Marktbedürfnisse reagieren und gewinnen einen Vorteil.
- [Release-Angst](release-angst.md)
<br/>  Seltene, große Releases werden zu Hochrisiko-Ereignissen, die Stress und Angst rund um jedes Deployment erzeugen.

## Causes ▼

- [Erhöhter manueller Testaufwand](erhoehter-manueller-testaufwand.md)
<br/>  Umfangreiche manuelle Testanforderungen für jedes Release verlängern direkt die Dauer des Release-Zyklus.
- [Manuelle Deployment-Prozesse](manuelle-deployment-prozesse.md)
<br/>  Manuelle Deployment-Verfahren fügen jedem Release Overhead und Koordinationszeit hinzu, was häufige Releases entmutigt.
- [Schlechte Testabdeckung](schlechte-testabdeckung.md)
<br/>  Ohne angemessene automatisierte Tests müssen Teams sich auf langwierige manuelle Testphasen verlassen, um Releases zu validieren.
- [Lange Build- und Testzeiten](lange-build-und-testzeiten.md)
<br/>  Langsame Build- und Test-Pipelines verlängern die Feedback-Schleife, was häufigeres Releasen unpraktikabel macht.
- [Fehlende Rollback-Strategie](fehlende-rollback-strategie.md)
<br/>  Ohne Rollback-Fähigkeiten bündeln Teams mehr Änderungen in weniger Releases, um das Risiko unumkehrbarer Fehlschläge zu minimieren.

## Detection Methods ○
- **Release-Häufigkeits-Metriken:** Nachverfolgung der Zeit zwischen Releases und Vergleich mit Branchenstandards oder Zielen
- **Release-Vorbereitungszeit:** Messung, wie lange Teams mit der Vorbereitung jedes Releases verbringen
- **Timing der Fehlerentdeckung:** Überwachung, wann Fehler im Release-Zyklus gefunden werden (späte Entdeckung deutet auf Prozessprobleme hin)
- **Feature-Lieferzeit:** Nachverfolgung, wie lange Features von der Fertigstellung bis zur Nutzerverfügbarkeit brauchen
- **Release-Größenanalyse:** Messung der Menge an Code oder Anzahl der Features pro Release

## Examples

Ein Softwareunternehmen veröffentlicht Updates alle sechs Monate, weil jedes Release vier Wochen manuelles Testen über verschiedene Browser, Betriebssysteme und Gerätekonfigurationen erfordert. Während der Testphase entdecken sie typischerweise 20-30 Fehler, die Korrekturen erfordern, die dann zusätzliches Testen benötigen, was den Zyklus weiter verlängert. Wenn ein Release bereit ist, enthält es Änderungen von sechs Monaten, was es extrem schwierig macht, die Grundursache auftretender Probleme zu identifizieren. Nutzer fordern häufig Features oder Fehlerbehebungen an, müssen aber Monate warten, um sie zu erhalten. Ein weiteres Beispiel betrifft eine Finanzdienstleistungsanwendung, bei der regulatorische Compliance umfangreiche Dokumentations- und Genehmigungsprozesse für jedes Release erfordert. Das Unternehmen bündelt Änderungen in vierteljährliche Releases, um den Overhead der Compliance-Prozesse zu minimieren, aber dies bedeutet, dass kritische Sicherheitspatches oder von Nutzern angeforderte Features bis zu vier Monate brauchen können, um in Produktion zu gelangen, was Geschäftsrisiko und Nutzerunzufriedenheit schafft.
