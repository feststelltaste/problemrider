---
title: Single Points of Failure
description: Fortschritt wird blockiert, wenn spezifische Wissensträger oder Systemkomponenten
  nicht verfügbar sind, was kritische Abhängigkeiten schafft.
category:
- Management
- Process
- Team
related_problems:
- slug: knowledge-dependency
  similarity: 0.7
- slug: knowledge-silos
  similarity: 0.7
- slug: maintenance-bottlenecks
  similarity: 0.65
- slug: knowledge-gaps
  similarity: 0.65
- slug: bottleneck-formation
  similarity: 0.65
- slug: approval-dependencies
  similarity: 0.65
solutions:
- event-driven-architecture
- observability-and-monitoring
- bulkhead
- chaos-engineering
- circuit-breaker
- data-replication
- disaster-recovery
- distributed-processing
- failover-cluster
- failover-mechanisms
- fault-containment
- health-check-endpoints
- heartbeat
- high-availability-architectures
- horizontal-scaling
- isolation-of-faulty-components
- load-balancing
- nonstop-forwarding
- ping
- read-replicas
- redundancy
- redundant-data-storage
- regular-backups
- resilience
- security-architecture-analysis
- watchdog
- knowledge-rotation
- risk-quantification
- cost-of-delay
layout: problem
lang: de
en_slug: single-points-of-failure
---

## Description

Single Points of Failure treten auf, wenn kritisches Systemwissen, Fähigkeiten oder Prozesse vollständig von einzelnen Teammitgliedern oder spezifischen Systemkomponenten abhängen. Wenn diese Personen nicht verfügbar sind oder wenn Schlüsselkomponenten ausfallen, können ganze Projekte blockiert werden, kritische Probleme können nicht gelöst werden, und der Entwicklungsfortschritt stoppt. Dies schafft erhebliches organisatorisches Risiko und verringert die Teamresilienz, was die Organisation anfällig für Störungen durch Personalwechsel oder Systemausfälle macht.

## Indicators ⟡

- Bestimmte Teammitglieder sind essentiell für bestimmte Arten von Arbeit
- Die Entwicklung stoppt, wenn Schlüsselpersonen nicht verfügbar sind
- Kritische Systemkomponenten haben kein Backup oder keine Redundanz
- Bestimmte Probleme können nur von einer Person gelöst werden
- Das Team gerät in Panik, wenn Schlüsselpersonal krank oder im Urlaub ist

## Symptoms ▲

- [Langsame Vorfallslösung](langsame-vorfallsloesung.md)
<br/>  Wenn der einzige Experte nicht verfügbar ist, dauert die Lösung von Vorfällen viel länger, weil niemand sonst das erforderliche Wissen hat.
- [Langsame Entwicklungsgeschwindigkeit](langsame-entwicklungsgeschwindigkeit.md)
<br/>  Die Entwicklung stockt, wenn Schlüsselpersonen nicht verfügbar sind, was den Gesamtdurchsatz des Teams verringert.
- [Kaskadierende Verzögerungen](kaskadierende-verzoegerungen.md)
<br/>  Wenn ein Single Point of Failure nicht verfügbar wird, kaskadieren abhängige Arbeitselemente in Verzögerungen über mehrere Teams und Projekte hinweg.
- [Probleme mit der Personalverfügbarkeit](probleme-mit-der-personalverfuegbarkeit.md)
<br/>  Wenn kritische Arbeit von bestimmten Personen abhängt, schafft deren Nichtverfügbarkeit effektive Personallücken, selbst wenn das Team ansonsten vollständig besetzt ist.

## Causes ▼

- [Wissenssilos](wissenssilos.md)
<br/>  Wenn Wissen bei Einzelpersonen statt im gesamten Team geteilt konzentriert ist, werden diese Personen zu Single Points of Failure.
- [Wissensabhängigkeit](wissensabhaengigkeit.md)
<br/>  Kritisches Systemwissen, das bei bestimmten Personen liegt, schafft Abhängigkeiten, die sie unersetzlich machen.
- [Schlechte Dokumentation](schlechte-dokumentation.md)
<br/>  Ohne Dokumentation bleibt Wissen in den Köpfen Einzelner eingeschlossen, was die Organisation von deren Verfügbarkeit abhängig macht.
- [Lücken in der Kompetenzentwicklung](luecken-in-der-kompetenzentwicklung.md)
<br/>  Wenn Teammitglieder es versäumen, eine Bandbreite an Fähigkeiten zu entwickeln, konzentriert sich Expertise auf wenige Personen, die zu Single Points of Failure werden.
- [Unzureichendes Onboarding](unzureichendes-onboarding.md)
<br/>  Unzureichendes Onboarding verhindert, dass neue Teammitglieder das Wissen entwickeln, um als Backup für kritische Funktionen zu dienen.

## Detection Methods ○

- **Bus-Faktor-Analyse:** Identifikation, was passieren würde, wenn Schlüsselpersonen nicht verfügbar wären
- **Abhängigkeits-Mapping:** Kartierung, welche Arbeit von bestimmten Personen oder Systemen abhängt
- **Bewertung der Wissensverteilung:** Bewertung, wie gleichmäßig kritisches Wissen verteilt ist
- **Verfolgung der Verfügbarkeitsauswirkung:** Überwachung, wie oft individuelle Nichtverfügbarkeit Arbeit blockiert
- **Cross-Training-Audit:** Bewertung, wie viele Personen kritische Aufgaben durchführen können

## Examples

Der gesamte Deployment-Prozess hängt von einem Senior-Entwickler ab, der die komplexe Abfolge manueller Schritte, Serverkonfigurationen und Fehlerbehebungsprozeduren kennt. Wenn er eine Woche krank ist, sind Releases vollständig blockiert, weil niemand sonst versteht, wie man die Anwendung sicher deployt oder Deployment-Probleme behebt. Das Team entdeckt, dass es keine Dokumentation des Deployment-Prozesses gibt und dass Versuche anderer, zu deployen, in Systemausfällen resultieren. Ein weiteres Beispiel betrifft ein Legacy-Datenbanksystem, bei dem nur ein Teammitglied die komplexen Datenmigrationsskripte und Performance-Tuning-Prozeduren versteht. Wenn diese Person das Unternehmen verlässt, sieht sich das Team einer Krise gegenüber, weil kritische Datenbankwartungsaufgaben nicht mehr durchgeführt werden können und neue Features, die Datenbankänderungen erfordern, auf unbestimmte Zeit blockiert sind.
