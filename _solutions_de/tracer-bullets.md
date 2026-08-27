---
title: Tracer Bullets
description: Frühzeitige Validierung durchgängiger Funktionalität durch
  vereinfachte Implementierungen.
category:
- Architecture
- Process
problems:
- integration-difficulties
- implementation-starts-without-design
- assumption-based-development
- fear-of-change
- modernization-strategy-paralysis
- missing-end-to-end-tests
- complex-implementation-paths
- system-integration-blindness
layout: solution
lang: de
en_slug: tracer-bullets
related_solutions:
- slug: walking-skeleton
  similarity: 0.7
- slug: prototypes
  similarity: 0.7
- slug: functional-spike
  similarity: 0.7
- slug: prototyping
  similarity: 0.7
- slug: strangler-fig-pattern
  similarity: 0.65
- slug: architecture-decision-records
  similarity: 0.65
---

## Description

Ein Tracer Bullet ist eine dünne, durchgängige Implementierung eines einzelnen repräsentativen Szenarios, das jede Schicht einer vorgeschlagenen Architektur durchläuft — UI, Geschäftslogik, Datenzugriff, Integration, Deployment und Monitoring —, bewusst als Produktionscode gebaut statt als wegwerfbarer Prototyp. Der Punkt ist nicht, ein fertiges Feature zu liefern, sondern früh eine echte Anfrage durch den gesamten beabsichtigten Technologie-Stack zu feuern, sodass Integrationsannahmen gegen die Realität getestet werden, statt bis viel später im Projekt theoretisch zu bleiben. Dies ist besonders wertvoll in der Legacy-Modernisierung, wo die riskantesten Unbekannten selten in den neuen Komponenten selbst liegen, sondern darin, wie sie tatsächlich mit den Authentifizierungseigenheiten, Verbindungspool-Grenzen, Datenformaten und der Netzwerktopologie des Legacy-Systems interagieren werden — Details, die fast nie vollständig dokumentiert sind und die erst zutage treten, sobald etwas Konkretes versucht, mit dem Legacy-System zu sprechen. Indem der Tracer Bullet eng gehalten wird, aber echt vollständig von Anfang bis Ende, kann ein Team Integrations- und Deployment-Probleme entdecken, während sie noch günstig zu beheben sind, statt nachdem Dutzende von Features auf unvalidierten architektonischen Annahmen gebaut wurden. Da die Implementierung erhalten und erweitert wird statt verworfen, wird der Tracer Bullet auch zum ersten funktionierenden Referenzpunkt für die Zielarchitektur, was Stakeholdern sichtbaren, deploybaren Fortschritt gibt und dem Team direkte Erfahrung mit dem neuen Stack, bevor Lieferdruck einsetzt.

## How to Apply ◆

> In der Legacy-Modernisierung validieren Tracer Bullets den gesamten technischen Stack durchgängig mit einer vereinfachten Implementierung, bevor in vollständige Feature-Entwicklung investiert wird.

- Wählen Sie ein einzelnes, repräsentatives Geschäftsszenario, das alle Schichten der vorgeschlagenen Architektur berührt — von der UI über Geschäftslogik, Datenzugriff und Integration mit Legacy-Systemen.
- Implementieren Sie dieses Szenario als dünne, funktionierende Scheibe, die den vollständigen Technologie-Stack ausübt, einschließlich Deployment-Pipelines, Monitoring und Produktionsinfrastruktur.
- Nutzen Sie den Tracer Bullet, um Integrationspunkte mit dem Legacy-System früh zu validieren — API-Kompatibilität, Datenformatannahmen, Authentifizierungsmechanismen und Netzwerkkonnektivität.
- Halten Sie die Implementierung bewusst einfach, um sich darauf zu konzentrieren, die Architektur zu beweisen, statt produktionsqualitative Features zu liefern.
- Behandeln Sie den Tracer Bullet als Produktionscode, der erweitert wird, im Gegensatz zu einem Prototyp, der verworfen wird — dies stellt sicher, dass architektonische Entscheidungen in einem realistischen Kontext getestet werden.
- Nutzen Sie das Tracer-Bullet-Deployment, um betriebliche Belange zu validieren: Kann das Team unabhängig vom Legacy-System deployen, funktioniert Monitoring, sind Alarme korrekt konfiguriert?
- Iterieren Sie über die Architektur basierend auf Tracer-Bullet-Befunden, bevor Sie auf zusätzliche Features erweitern.

## Tradeoffs ⇄

> Tracer Bullets bieten frühe architektonische Validierung, erfordern aber Disziplin, um den anfänglichen Umfang eng genug zu halten, um nützlich zu sein.

**Vorteile:**

- Bringt Integrationsprobleme mit Legacy-Systemen Wochen oder Monate zutage, bevor sie die vollständige Entwicklung entgleisen lassen würden, wenn sie am günstigsten zu beheben sind.
- Bietet ein funktionierendes Skelett, auf dem nachfolgende Features gebaut werden können, was das Risiko architektonischer Entscheidungen reduziert, die auf dem Papier gut aussehen, aber in der Praxis scheitern.
- Gibt dem Team praktische Erfahrung mit dem neuen Technologie-Stack in einem produktionsähnlichen Kontext, bevor sie unter Fristendruck liefern müssen.
- Schafft früh im Projekt ein deploybares Artefakt und baut Stakeholder-Vertrauen durch sichtbaren, greifbaren Fortschritt auf.

**Kosten und Risiken:**

- Wenn der Tracer-Bullet-Umfang zu ambitioniert ist, wird er zu einem Mini-Projekt, das die Entwicklung verzögert statt beschleunigt.
- Teams könnten versucht sein, den Tracer Bullet zu überspringen und direkt in die Feature-Entwicklung zu springen, besonders unter Lieferdruck.
- Die vereinfachte Implementierung bringt möglicherweise keine Probleme zutage, die sich nur unter Produktionsskala-Last oder Datenvolumina manifestieren.
- Tracer Bullets validieren die gewählte Architektur, können aber Commitment-Bias schaffen — das Team könnte widerwillig sein, architektonische Entscheidungen zu ändern, selbst wenn spätere Evidenz darauf hindeutet, dass sie es sollten.

## How It Could Be

> Das folgende Szenario veranschaulicht, wie ein Tracer Bullet Architektur in einem Legacy-Modernisierungskontext validiert.

Ein Telekommunikationsunternehmen modernisierte sein Kundenservice-Portal von einer Legacy-JSP-Anwendung mit Oracle-Datenbank-Backend zu einem React-Frontend mit Microservices. Bevor irgendwelche kundenseitigen Features gebaut wurden, implementierte das Team einen einzelnen Tracer Bullet: die Anzeige des aktuellen Kontostands eines Kunden. Dieses scheinbar einfache Feature erforderte, dass das neue React-Frontend ein neues API-Gateway aufrief, das zu einem neuen Konto-Service routete, der aus der Legacy-Oracle-Datenbank durch eine Anti-Corruption-Layer lesen musste. Der Tracer Bullet offenbarte drei kritische Probleme: die Verbindungspool-Konfiguration der Legacy-Datenbank konnte das Verbindungsmuster des neuen Dienstes nicht handhaben, die Timeout-Einstellungen des API-Gateways waren zu aggressiv für die Antwortzeiten der Legacy-Datenbank, und die Deployment-Pipeline konnte die Multi-Service-Deployment-Sequenz nicht handhaben. Die Behebung dieser Probleme dauerte zwei Wochen — wären sie während der vollständigen Entwicklung mit Dutzenden von Features in Arbeit entdeckt worden, wäre die Auswirkung Monate an Verzögerungen gewesen.
