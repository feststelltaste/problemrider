---
title: Kapazitäts-Fehlanpassung
description: Verfügbare Kapazität an unterschiedlichen Stufen des Entwicklungsprozesses
  passt nicht zu den Nachfragemustern, was Engpässe und Unterauslastung erzeugt.
category:
- Performance
- Process
related_problems:
- slug: bottleneck-formation
  similarity: 0.65
- slug: uneven-work-flow
  similarity: 0.65
- slug: work-queue-buildup
  similarity: 0.65
- slug: resource-waste
  similarity: 0.65
- slug: uneven-workload-distribution
  similarity: 0.65
- slug: staff-availability-issues
  similarity: 0.65
solutions:
- capacity-planning
- elastic-scaling
- distributed-processing
- elastic-resource-utilization
- failover-cluster
- graceful-degradation
- high-availability-architectures
- horizontal-scaling
- load-balancing
- load-shedding
- load-testing
- monitoring-system-utilization
- performance-measurements
- performance-modeling
- proactive-capacity-management
- rate-limiting
- redundancy
- serverless-computing
- specialized-hardware
- stress-testing
- vertical-scaling
layout: problem
lang: de
en_slug: capacity-mismatch
---

## Description

Kapazitäts-Fehlanpassung entsteht, wenn die verfügbare Kapazität an unterschiedlichen Stufen des Entwicklungsprozesses nicht mit den tatsächlichen Nachfragemustern übereinstimmt. Dies kann sich so äußern, dass manche Prozessstufen überlastet sind, während andere überschüssige Kapazität haben, was einen ineffizienten Fluss erzeugt, bei dem Ressourcen verschwendet werden, während sich anderswo Engpässe bilden. Effektive Kapazitätsabstimmung erfordert das Verständnis von Nachfragemustern und die entsprechende Zuweisung von Ressourcen.

## Indicators ⟡

- Manche Teammitglieder oder Prozessstufen sind durchgängig überlastet, während andere freie Kapazität haben
- Arbeitswarteschlangen stauen sich an bestimmten Stufen an, während andere Stufen untätig bleiben
- Der Prozessdurchsatz ist durch Kapazitätsengpässe an bestimmten Stufen begrenzt
- Die Ressourcenauslastung variiert dramatisch über verschiedene Rollen oder Prozessschritte hinweg
- Das Hinzufügen von Ressourcen zum Team verbessert den Gesamtdurchsatz nicht

## Symptoms ▲

- [Engpassbildung](engpassbildung.md)
<br/>  Stufen mit unzureichender Kapazität werden zu Engpässen, die den Durchsatz des gesamten Entwicklungsprozesses einschränken.
- [Ressourcenverschwendung](ressourcenverschwendung.md)
<br/>  Teammitglieder an Stufen mit überschüssiger Kapazität sitzen untätig, während eingeschränkte Stufen überlastet sind.
- [Aufstauung von Arbeitswarteschlangen](aufstauung-von-arbeitswarteschlangen.md)
<br/>  Arbeit häuft sich an unterkapazitierten Stufen an, was wachsende Warteschlangen erzeugt, die die Lieferung verzögern.
- [Entwicklerfrustration und Burnout](entwicklerfrustration-und-burnout.md)
<br/>  Entwickler werden frustriert, wenn sie aufgrund fehlausgerichteter Kapazität über Stufen hinweg entweder überlastet oder untätig sind.
- [Kaskadierende Verzögerungen](kaskadierende-verzoegerungen.md)
<br/>  Kapazitätsbeschränkte Stufen verursachen Verzögerungen, die sich zu nachgelagerter Arbeit und abhängigen Projekten fortpflanzen.

## Causes ▼

- [Schlechte Planung](schlechte-planung.md)
<br/>  Das Versäumnis, Nachfragemuster zu analysieren, führt zu einer Kapazitätszuweisung, die nicht zu den tatsächlichen Workflow-Bedürfnissen passt.
- [Wissenssilos](wissenssilos.md)
<br/>  Auf wenige Personen konzentriertes Spezialwissen erzeugt Kapazitätsengpässe an Stufen, die diese Expertise erfordern.
- [Fehlpassung der Organisationsstruktur](fehlpassung-der-organisationsstruktur.md)
<br/>  Teamstrukturen, die nicht mit den Prozessanforderungen übereinstimmen, erzeugen inhärente Kapazitätsungleichgewichte über Workflow-Stufen hinweg.

## Detection Methods ○

- **Kapazitätsauslastungsanalyse:** Beobachtung der Auslastungsraten über verschiedene Rollen und Prozessstufen hinweg
- **Durchsatzanalyse:** Messung des tatsächlichen Durchsatzes gegenüber der theoretischen Kapazität an jeder Stufe
- **Warteschlangenlängen-Monitoring:** Nachverfolgung, wo sich Arbeit anhäuft und wo Kapazität untätig bleibt
- **Ressourcenzuweisungs-Review:** Bewertung, ob die Ressourcenverteilung zu den tatsächlichen Nachfragemustern passt
- **Flusseffizienz-Messung:** Berechnung der Gesamtprozesseffizienz unter Berücksichtigung von Kapazitätsengpässen

## Examples

Ein Softwareentwicklungsteam hat fünf Entwickler, aber nur eine Person, die für Datenbankarbeit qualifiziert ist. Das Team hat durchgängig datenbankbezogene Aufgaben, die sich in einer Warteschlange stauen, während andere Entwickler auf den Abschluss von Datenbankänderungen warten, bevor sie mit ihrer eigenen Arbeit fortfahren können. Trotz ausreichender Gesamtentwicklungskapazität ist der Durchsatz des Teams durch den einzigen Datenbankspezialisten begrenzt. Ein weiteres Beispiel betrifft ein Testteam, bei dem die Kapazität für automatisierte Testerstellung die Kapazität für manuelles Testen weit übersteigt, aber der Prozess eine manuelle Verifikation aller automatisierten Testergebnisse erfordert. Das Team für automatisiertes Testen wird untätig, während es darauf wartet, dass manuelle Tester aufholen, während das Team für manuelles Testen mit Verifikationsarbeit überlastet ist.
