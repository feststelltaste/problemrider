---
title: Fehler bei der Service Discovery
description: Service-Discovery-Mechanismen schaffen es nicht, Services zu lokalisieren
  oder mit ihnen zu verbinden, was Kommunikationsfehler und Systeminstabilität in
  verteilten Architekturen verursacht.
category:
- Architecture
- Operations
- Performance
related_problems:
- slug: service-timeouts
  similarity: 0.65
- slug: system-outages
  similarity: 0.65
- slug: cascade-failures
  similarity: 0.6
- slug: load-balancing-problems
  similarity: 0.6
- slug: configuration-chaos
  similarity: 0.55
- slug: upstream-timeouts
  similarity: 0.55
solutions:
- api-first-design
- elastic-scaling
- api-gateway
- health-check-endpoints
- ping
- service-mesh
- monitoring
- observability-and-monitoring
- circuit-breaker
- retry
- production-readiness-criteria
layout: problem
lang: de
en_slug: service-discovery-failures
---

## Description

Fehler bei der Service Discovery treten auf, wenn verteilte Systeme benötigte Services nicht zuverlässig lokalisieren und mit ihnen verbinden können, was zu Kommunikationszusammenbrüchen und Systeminstabilität führt. Dies betrifft Microservices-Architekturen, Cloud-native Anwendungen und jedes System, das auf dynamischer Service-Registrierung und -Erkennung basiert. Fehlgeschlagene Service Discovery kann sich durch abhängige Services kaskadieren und weitreichende Systemausfälle verursachen.

## Indicators ⟡

- Services melden „Service nicht gefunden"- oder Verbindungs-Timeout-Fehler
- Intermittierende Fehler in der Service-zu-Service-Kommunikation
- Load Balancer können Traffic nicht zu gesunden Service-Instanzen routen
- Die Service-Registry zeigt veraltete oder inkorrekte Service-Informationen
- Anwendungen können aufgrund fehlender Service-Abhängigkeiten nicht starten

## Symptoms ▲

- [Kaskadierende Ausfälle](kaskadierende-ausfaelle.md)
<br/>  Wenn Service Discovery fehlschlägt, können abhängige Services ihre Abhängigkeiten nicht lokalisieren, was Fehler durch das System kaskadieren lässt.
- [Systemausfälle](systemausfaelle.md)
<br/>  Weitreichende Fehler bei der Service Discovery können ganze verteilte Systeme lahmlegen, da Services die Fähigkeit zur Kommunikation verlieren.
- [Service-Timeouts](service-timeouts.md)
<br/>  Fehlgeschlagene Service Discovery verursacht, dass Services Verbindungsversuche zu veralteten oder ungültigen Endpunkten unternehmen, was zu Verbindungs-Timeouts führt.
- [Erhöhte Fehlerraten](erhoehte-fehlerraten.md)
<br/>  Services, die ihre Abhängigkeiten nicht entdecken können, erzeugen Verbindungsfehler und „Service nicht gefunden"-Fehler in erhöhter Rate.
- [Langsame Vorfallslösung](langsame-vorfallsloesung.md)
<br/>  Fehler bei der Service Discovery sind schwer zu diagnostizieren, weil sie sich als verschiedenste nachgelagerte Symptome äußern, was die Ursachenidentifikation verlangsamt.

## Causes ▼

- [Netzwerklatenz](netzwerklatenz.md)
<br/>  Hohe Netzwerklatenz verursacht Timeouts bei Service-Registrierung und Health-Checks, was die Discovery-Registry mit veralteten Informationen zurücklässt.
- [Konfigurations-Drift](konfigurations-drift.md)
<br/>  Service-Discovery-Konfigurationen driften über Umgebungen hinweg auseinander, was inkonsistentes Verhalten bei Service-Registrierung und -Auflösung verursacht.
- [Schlechte Systemumgebung](schlechte-systemumgebung.md)
<br/>  Instabile Infrastruktur, die Service-Discovery-Komponenten hostet, führt zu intermittierenden Fehlern bei Service-Registrierung und -Auflösung.

## Detection Methods ○

- **Service-Discovery-Monitoring:** Überwachung der Gesundheit und Antwortzeiten der Service-Registry
- **Service-Auflösungstests:** Regelmäßiges Testen der Service-Namensauflösung im gesamten System
- **Health-Check-Validierung:** Verifikation, dass Health-Checks den Service-Status akkurat widerspiegeln
- **Netzwerkverbindungstests:** Testen von Netzwerkpfaden zwischen Services und Discovery-Infrastruktur
- **Registry-Konsistenzauditierung:** Auditierung von Service-Registry-Daten auf Konsistenz und Genauigkeit

## Examples

Eine Microservices-E-Commerce-Plattform nutzt Consul für Service Discovery, aber Netzwerklatenz verursacht Timeouts bei Service-Registrierungen, was die Registry mit veralteten Informationen zurücklässt. Wenn Zahlungsservices versuchen, sich mit Bestandsservices zu verbinden, erhalten sie veraltete IP-Adressen beendeter Instanzen, was Checkout-Fehler verursacht. Die Implementierung von Retry-Logik und Verbesserungen der Health-Checks löst die Zuverlässigkeitsprobleme der Discovery. Ein weiteres Beispiel betrifft einen Kubernetes-Cluster, bei dem DNS-basierte Service Discovery intermittierend aufgrund von DNS-Server-Überlastung fehlschlägt. Anwendungen erleben zufällige Verbindungsfehler bei der Auflösung von Service-Namen, besonders während Zeiten hohen Traffics. Die Skalierung der DNS-Infrastruktur und die Implementierung von DNS-Caching reduziert Discovery-Fehler um 95 %.
