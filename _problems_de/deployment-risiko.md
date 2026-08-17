---
title: Deployment-Risiko
description: System-Deployments bergen aufgrund irreversibler Änderungen und fehlender
  Wiederherstellungsmechanismen ein hohes Ausfall- oder Schadensrisiko.
category:
- Management
- Operations
- Process
related_problems:
- slug: missing-rollback-strategy
  similarity: 0.8
- slug: large-risky-releases
  similarity: 0.75
- slug: complex-deployment-process
  similarity: 0.75
- slug: manual-deployment-processes
  similarity: 0.7
- slug: immature-delivery-strategy
  similarity: 0.65
- slug: history-of-failed-changes
  similarity: 0.65
solutions:
- blue-green-canary-deployments
- ci-cd-pipeline
- feature-flags
- infrastructure-as-code
- backup-and-recovery
- backward-compatibility
- backward-compatible-schema-migrations
- canary-releases
- containerization
- continuous-delivery
- continuous-integration
- continuous-integration-and-delivery
- dark-launches
- dependency-pinning
- disaster-recovery
- environment-parity
- failover-cluster
- feature-toggles
- immutable-infrastructure
- integration-tests
- load-testing
- redundancy
- regular-backups
- restore-points
- risk-analysis
- rollback-mechanisms
- rolling-updates
- site-reliability-engineering-sre
- smoke-testing
- standardized-deployment-scripts
- continuous-deployment
- digital-signatures
- error-budgets
- malware-protection
- patch-management
- self-test
- risk-quantification
- baseline-measurement
layout: problem
lang: de
en_slug: deployment-risk
---

## Description

Deployment-Risiko entsteht, wenn die Veröffentlichung von Softwareänderungen eine hohe Wahrscheinlichkeit birgt, Systemausfälle, Datenverlust oder verlängerte Ausfallzeiten zu verursachen, mit begrenzter Fähigkeit zur schnellen Wiederherstellung. Dieses Risiko äußert sich, wenn Deployment-Prozesse irreversible Änderungen vornehmen, keine getesteten Wiederherstellungsmechanismen haben oder komplexe manuelle Eingriffe erfordern, die fehlschlagen können. Hohes Deployment-Risiko schafft einen Kreislauf, in dem Teams selten deployen, um das Risiko zu minimieren, aber seltene Deployments machen jedes Release größer und riskanter.

## Indicators ⟡

- Deployments erfordern umfangreiche Planung und mehrere Teammitglieder
- Das Team plant Deployments aufgrund erwarteter Probleme außerhalb der Geschäftszeiten
- Datenbankmigrationen oder Schemaänderungen verursachen besondere Angst
- Die Wiederherstellung nach Deployment-Problemen erfordert Stunden oder manuelles Eingreifen
- Deployments werden aufgrund von Risikobedenken verschoben oder vermieden

## Symptoms ▲

- [Lange Release-Zyklen](lange-release-zyklen.md)
<br/>  Teams deployen selten, um das Risiko zu minimieren, aber dies verlängert die Release-Zyklen.
- [Große, riskante Releases](grosse-riskante-releases.md)
<br/>  Seltenes Deployment aufgrund hohen Risikos führt dazu, dass sich Änderungen zu großen, noch riskanteren Bündeln anhäufen.
- [Release-Angst](release-angst.md)
<br/>  Hohes Deployment-Risiko erzeugt erhebliche Angst und Stress für Teams, die für Releases verantwortlich sind.
- [Angst vor Veränderung](angst-vor-veraenderung.md)
<br/>  Wenn Deployments riskant sind, werden Teams zurückhaltend, Änderungen vorzunehmen, was zu Systemstagnation führt.
- [Verzögerte Wertlieferung](verzoegerte-wertlieferung.md)
<br/>  Die Angst vor riskanten Deployments verzögert es, fertiggestellte Features in die Produktion und zu Nutzern zu bringen.
- [Systemausfälle](systemausfaelle.md)
<br/>  Riskante Deployments, die schiefgehen, können aufgrund fehlender Wiederherstellungsmechanismen verlängerte Ausfälle verursachen.

## Causes ▼

- [Manuelle Deployment-Prozesse](manuelle-deployment-prozesse.md)
<br/>  Manuelle Deployment-Schritte sind fehleranfällig und haben nicht die Sicherheitsnetze, die automatisierte Prozesse bieten.
- [Deployment-Kopplung](deployment-kopplung.md)
<br/>  Gekoppelte Deployments erfordern die Koordination mehrerer Komponenten, was die Wahrscheinlichkeit erhöht, dass etwas schiefgeht.
- [Inkonsistenzen zwischen Deployment-Umgebungen](inkonsistenzen-zwischen-deployment-umgebungen.md)
<br/>  Umgebungsunterschiede bedeuten, dass Tests das Produktionsverhalten nicht garantieren können, was das Deployment-Risiko erhöht.
- [Schlechte Testabdeckung](schlechte-testabdeckung.md)
<br/>  Ohne umfassende Tests gibt es wenig Vertrauen darauf, dass Änderungen bestehende Funktionalität während des Deployments nicht brechen.
- [Komplexer Deployment-Prozess](komplexer-deployment-prozess.md)
<br/>  Komplexe, mehrstufige Deployment-Prozesse haben mehr Fehlerpunkte und sind schwerer korrekt auszuführen.

## Detection Methods ○

- **Deployment-Erfolgsrate:** Nachverfolgung des Prozentsatzes von Deployments, die ohne Probleme abgeschlossen werden
- **Wiederherstellungszeit-Analyse:** Messung der Zeit, die nötig ist, um Deployment-Probleme zu lösen
- **Deployment-Häufigkeit vs. Risiko:** Analyse der Korrelation zwischen Deployment-Häufigkeit und Problemen
- **Bewertung der Rollback-Fähigkeit:** Bewertung der Fähigkeit, problematische Deployments schnell zurückzunehmen
- **Deployment-Prozesskomplexität:** Nachverfolgung der Anzahl manueller Schritte und potenzieller Fehlerpunkte
- **Team-Stress-Indikatoren:** Beobachtung von Teamangst und Überstunden im Zusammenhang mit Deployments

## Examples

Eine Finanzdienstleistungsanwendung erfordert für jedes Release Datenbankschemaänderungen, und diese Migrationen können mehrere Stunden dauern, während derer das System nicht verfügbar ist. Wenn eine Migration mittendrin fehlschlägt, bleibt die Datenbank in einem inkonsistenten Zustand zurück, der manuelles Eingreifen durch Datenbankadministratoren erfordert, was möglicherweise verlängerte Ausfälle verursacht. Das Team deployt aufgrund dieses Risikos nur einmal im Monat, aber monatliche Releases sind groß und komplex, was Fehlschläge wahrscheinlicher macht. Ein weiteres Beispiel betrifft eine Microservices-Plattform, bei der Deployments koordinierte Aktualisierungen über mehrere Services in einer bestimmten Reihenfolge erfordern. Wenn ein Service nicht korrekt deployt wird, kann das gesamte System instabil werden, aber ein Rollback erfordert das manuelle Zurücknehmen jedes Service in umgekehrter Reihenfolge, ein Prozess, der oft zusätzliche Fehler einführt und den Ausfall verlängert.
