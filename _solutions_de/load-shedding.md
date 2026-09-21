---
title: Load Shedding
description: Gezieltes Verwerfen niedrig priorisierter Anfragen bei Überlast,
  um kritische Kapazität zu erhalten.
category:
- Architecture
- Performance
problems:
- capacity-mismatch
- slow-application-performance
- system-outages
- cascade-failures
- rate-limiting-issues
- task-queues-backing-up
- unbounded-data-structures
- insufficient-worker-capacity
- work-queue-buildup
layout: solution
lang: de
en_slug: load-shedding
related_solutions:
- slug: rate-limiting
  similarity: 0.8
- slug: graceful-degradation
  similarity: 0.75
- slug: load-balancing
  similarity: 0.7
- slug: backpressure
  similarity: 0.7
- slug: distributed-caching
  similarity: 0.65
- slug: lazy-loading
  similarity: 0.65
---

## Description

Load Shedding ist die bewusste, kontrollierte Ablehnung eines Teils des eingehenden Verkehrs, wenn ein System überlastet ist, sodass die verbleibende Kapazität für die wichtigsten Anfragen erhalten bleibt, statt so dünn über alle Anfragen verteilt zu werden, dass alles ausfällt. Es funktioniert, indem Anfragen im Voraus in Prioritätsstufen klassifiziert werden, aktuelle Last gegen definierte Schwellenwerte gemessen wird und das System niedrigpriorisierte Arbeit aktiv ablehnt oder aufschiebt — typischerweise mit einem expliziten Status wie 503 mit einem Retry-Hinweis —, während kritische Pfade wie Authentifizierung oder Zahlung weiterhin normal bedient werden. Legacy-Systeme sind besonders anfällig für Überlastkollaps, weil sie oft ganz ohne Zulassungskontrolle gebaut wurden: Jede Anfrage wird identisch behandelt, Ressourcen werden nach dem First-Come-Prinzip verbraucht, und sobald die Nachfrage die Kapazität übersteigt, degradiert das System nicht sanft, sondern kommt für jeden Nutzer gleichzeitig zum Stillstand, einschließlich derjenigen, die die geschäftskritischsten Aktionen durchführen. Load Shedding einzuführen verwandelt einen unkontrollierten Alles-oder-nichts-Ausfallmodus in einen entworfenen, partiellen, was eine bedeutende Verbesserung für Legacy-Anwendungen ist, die nicht leicht für elastische Skalierung neu entworfen werden können und stattdessen Nachfragespitzen mit ungefähr fester Kapazität überstehen müssen. Der Ansatz hängt von einem genauen, kontinuierlich gepflegten Verständnis ab, welche Anfragen aus geschäftlicher Perspektive tatsächlich niedrige Priorität haben, was oft der schwierigste Teil ist, in einem Legacy-System zu etablieren, wo diese Klassifizierung nie von vornherein explizit gemacht wurde.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken, um diese Lösung im Kontext eines Legacy-Systems umzusetzen.

- Klassifizieren Sie alle Anfragetypen im Legacy-System nach geschäftlicher Priorität (kritisch, wichtig, Best-Effort)
- Implementieren Sie Zulassungskontrolle, die aktuelle Systemlast misst und niedrigpriorisierte Anfragen ablehnt, wenn Schwellenwerte überschritten werden
- Geben Sie angemessene HTTP-Statuscodes zurück (503 mit Retry-After), damit Clients zurückweichen und erneut versuchen können
- Stellen Sie sicher, dass kritische Pfade wie Zahlungen, Authentifizierung und Kerntransaktionen immer zuerst bedient werden
- Konfigurieren Sie warteschlangenbasierte Systeme, um niedrigpriorisierte Nachrichten zu verwerfen oder aufzuschieben, wenn die Queue-Tiefe Grenzen überschreitet
- Überwachen Sie das Volumen verworfener Last und alarmieren Sie, wenn die Häufigkeit des Verwerfens auf einen Bedarf für Kapazitätserweiterung hindeutet

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie dafür aufgeben.

**Vorteile:**
- Hält kritische Systemfunktionen während Überlastsituationen verfügbar
- Verhindert vollständigen Systemzusammenbruch durch proaktives Nachfragemanagement
- Bietet eine kontrollierte Reaktion auf Verkehrsspitzen statt unvorhersehbarer Ausfälle
- Verschafft Zeit für Auto-Scaling oder manuelles Eingreifen

**Kosten und Risiken:**
- Verworfene Anfragen verschlechtern die Nutzererfahrung für niedrigpriorisierte Operationen
- Prioritätsklassifizierung erfordert sorgfältigen geschäftlichen Input und laufende Pflege
- Falsche Prioritätszuweisungen können wichtigen Verkehr verwerfen
- Legacy-Systemen könnte die Instrumentierung fehlen, die nötig ist, um Last genau zu messen

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Eine auf einem Legacy-Stack gebaute Ticketverkaufsplattform erlebte vollständige Ausfälle während Hochnachfrage-Events, wenn alle Nutzer um begrenztes Inventar konkurrierten. Das Team implementierte Load Shedding, das Checkout- und Zahlungsanfragen priorisierte, während Such- und Browsing-Anfragen abgelehnt oder in eine Warteschlange gestellt wurden, wenn die Systemlast 80 % Kapazität überschritt. Beim nächsten großen Verkaufsevent blieb der Checkout-Flow reaktionsschnell, während manche Nutzer vorübergehende Verzögerungen bei Suchergebnissen erlebten. Die Gesamtzahl erfolgreicher Transaktionen stieg um 35 % im Vergleich zu vorherigen Events, bei denen das gesamte System unter der Last zusammengebrochen war.
