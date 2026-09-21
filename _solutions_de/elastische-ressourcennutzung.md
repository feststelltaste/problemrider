---
title: Elastische Ressourcennutzung
description: Automatische Anpassung von Ressourcen basierend auf aktueller Last.
category:
- Operations
- Performance
problems:
- scaling-inefficiencies
- capacity-mismatch
- slow-application-performance
- system-outages
- resource-contention
- high-database-resource-utilization
- resource-allocation-failures
layout: solution
lang: de
en_slug: elastic-resource-utilization
related_solutions:
- slug: elastic-scaling
  similarity: 0.8
- slug: horizontal-scaling
  similarity: 0.8
- slug: monitoring-system-utilization
  similarity: 0.75
- slug: load-balancing
  similarity: 0.75
- slug: cloud-native-development
  similarity: 0.75
- slug: proactive-capacity-management
  similarity: 0.75
---

## Description

Elastische Ressourcennutzung passt die einem System zugewiesenen Rechenressourcen automatisch als Reaktion auf Echtzeit-Last an, skaliert Kapazität aus, wenn die Nachfrage steigt, und zurück, wenn sie sinkt, statt auf einer festen Hardware-Menge zu laufen, dimensioniert entweder für den Durchschnittsfall oder, schlimmer, für den Worst Case, der nur selten eintritt. Legacy-Systeme sind häufig genau auf dieser Art fester, statisch bereitgestellter Hardware deployt, was bedeutet, dass sie entweder überprovisioniert und verschwenderisch untätig die meiste Zeit dasitzen oder komplett versagen in dem Moment, in dem der Traffic welche Kapazität auch immer ursprünglich geplant wurde übersteigt — eine Diskrepanz, die besonders während unvorhersehbarer Nachfragespitzen sichtbar wird, die die ursprüngliche Architektur nie antizipierte. Elastizität zu erreichen erfordert typischerweise, dass die Legacy-Anwendung zuerst horizontal skalierbar wird, was bedeutet, Sitzungszustand zu externalisieren, das Deployment zu containerisieren und die Last- und Performance-Kennzahlen bereitzustellen, die eine Auto-Scaling-Richtlinie für ihre Entscheidungen braucht — Arbeit, die für Anwendungen, die ursprünglich mit der Annahme eines einzelnen, dauerhaften Servers gebaut wurden, nicht trivial ist. Einmal etabliert, entfernt dies manuelle Kapazitätsplanung als Engpass für variable Workloads und lässt Infrastrukturausgaben tatsächliche Nutzung statt Worst-Case-Spitzen verfolgen, was oft Kosten senkt, während sich gleichzeitig die Zuverlässigkeit verbessert. Die Tradeoffs sind echt: Scale-Out-Verzögerung kann während einer sehr plötzlichen Spitze einen kurzen Performance-Einbruch hinterlassen, falsch konfigurierte Richtlinien können entweder überausgeben oder unterversorgen, und das resultierende verteilte, dynamisch dimensionierte Deployment ist inhärent schwerer zu überwachen und zu debuggen als der einzelne feste Server, den es ersetzte.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken zur Implementierung dieser Lösung im Kontext eines Legacy-Systems.

- Instrumentieren Sie die Anwendung, um Kennzahlen bereitzustellen, die Skalierungsentscheidungen antreiben (CPU, Speicher, Anfragewarteschlangentiefe, Antwortzeit)
- Containerisieren Sie die Legacy-Anwendung oder deployen Sie sie hinter einem Load Balancer, der dynamische Backend-Registrierung unterstützt
- Konfigurieren Sie Auto-Scaling-Richtlinien basierend auf beobachteten Traffic-Mustern und Performance-Schwellenwerten
- Definieren Sie Mindest- und Höchstressourcengrenzen, um unkontrollierte Skalierung zu verhindern und Kosten zu kontrollieren
- Implementieren Sie Health Checks, die Auto-Scaling-Systeme nutzen, um die Bereitschaft einer Instanz zu bestimmen, bevor Traffic geroutet wird
- Gestalten Sie die Anwendung zustandslos oder nutzen Sie externalisierten Sitzungsspeicher, damit Instanzen frei hinzugefügt und entfernt werden können
- Testen Sie das Skalierungsverhalten unter Last, um zu verifizieren, dass Scale-Out und Scale-In korrekt funktionieren, ohne Anfragen zu verlieren

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie aufgeben.

**Vorteile:**
- Handhabt Traffic-Spitzen automatisch, ohne manuellen Eingriff oder Überprovisionierung
- Reduziert Kosten während Zeiten geringen Traffics, indem ungenutzte Ressourcen heruntergeskaliert werden
- Verbessert die Systemzuverlässigkeit, indem Last über mehrere Instanzen verteilt wird
- Eliminiert Kapazitätsplanungs-Rätselraten für variable Workloads

**Kosten und Risiken:**
- Legacy-Anwendungen mit zustandsbehaftetem Design erfordern Refactoring, bevor sie horizontal skalieren können
- Auto-Scaling-Verzögerung kann während plötzlicher Traffic-Spitzen kurze Performance-Verschlechterung verursachen
- Falsch konfigurierte Skalierungsrichtlinien können zu exzessiven Kosten oder unzureichenden Ressourcen führen
- Cold-Start-Zeiten für neue Instanzen könnten für latenzsensitive Anwendungen zu langsam sein
- Erhöhte Komplexität bei der Überwachung und Fehlersuche verteilter Instanzen

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Eine Ticketverkaufsplattform erlebte extreme Traffic-Spitzen während beliebter Event-Veröffentlichungen, wobei die Last innerhalb von Minuten um das 50-Fache stieg. Der Legacy-Monolith war auf Hardware fester Größe deployt, die diese Spitzen nicht handhaben konnte, was Ausfälle in den kritischsten Geschäftsmomenten zur Folge hatte. Das Team containerisierte die Anwendung mit Docker, externalisierte Sitzungszustand nach Redis und deployte auf Kubernetes mit horizontaler Pod-Autoskalierung basierend auf Anfragewarteschlangentiefe. Während der nächsten großen Ticket-Veröffentlichung skalierte das System automatisch von 4 auf 60 Pods innerhalb von drei Minuten, handhabte den Spitzentraffic ohne Verschlechterung und skalierte innerhalb einer Stunde zurück. Die Infrastrukturkosten sanken tatsächlich, weil sie nicht mehr rund um die Uhr Hardware mit Spitzenkapazität vorhalten mussten.
