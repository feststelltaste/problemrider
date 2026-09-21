---
title: Retry
description: Wiederholung fehlgeschlagener Operationen zur Behandlung
  vorübergehender Fehler.
category:
- Code
- Architecture
problems:
- service-timeouts
- cascade-failures
- inadequate-error-handling
- unpredictable-system-behavior
- external-service-delays
- increased-error-rates
- upstream-timeouts
- service-discovery-failures
layout: solution
lang: de
en_slug: retry
related_solutions:
- slug: failover-mechanisms
  similarity: 0.8
- slug: rate-limiting
  similarity: 0.8
- slug: circuit-breaker
  similarity: 0.8
- slug: resilience
  similarity: 0.8
- slug: error-handling
  similarity: 0.75
- slug: chaos-engineering
  similarity: 0.75
---

## Description

Retry ist die Praxis, eine Operation, die aufgrund eines vorübergehenden Zustands fehlgeschlagen ist — eine kurze Netzwerkunterbrechung, eine kurzzeitig nicht verfügbare Abhängigkeit, ein temporäres Timeout —, automatisch erneut zu versuchen, statt den Fehler dem Nutzer offenzulegen oder sofort manuellen Eingriff zu erfordern. Effektive Retry-Logik unterscheidet zwischen Fehlern, die eine Wiederholung wert sind, wie Verbindungs-Timeouts, und Fehlern, die nie gelingen werden, egal wie oft sie wiederholt werden, wie Authentifizierungsfehler oder Validierungsfehler, und sie staffelt wiederholte Versuche mittels exponentiellem Backoff und Jitter, um zu vermeiden, eine bereits kämpfende Abhängigkeit zu überwältigen. In Legacy-Systemen sind Integrationspunkte mit externen Diensten oder zwischen intern zerlegten Komponenten häufig der unzuverlässigste Teil der Architektur, über die Jahre inkrementell hinzugefügt, ohne die Resilienzmuster, die in einem heute entworfenen System als Standard gelten würden; Retry ist einer der günstigsten Wege, diese Lücke zu schließen, da es üblicherweise um einen bestehenden Aufruf herum hinzugefügt werden kann, ohne die zugrunde liegende Operation zu modifizieren. Es ist besonders effektiv darin, die Klasse von Fehlern zu beseitigen, die zuvor erforderte, dass ein Mensch eine fehlgeschlagene Anfrage bemerkt, diagnostiziert und manuell erneut einreicht, was in Legacy-Betriebsumgebungen oft unverhältnismäßigen Support-Aufwand für Probleme verbrauchte, die sich innerhalb von Sekunden selbst lösten. Die Technik trägt jedoch eine spezifische Gefahr in Legacy-Kontexten: Viele ältere Operationen wurden nie idempotent entworfen, sodass blindes Wiederholen doppelte Transaktionen oder Seiteneffekte produzieren kann, weshalb Retry mit einer expliziten Prüfung — oder einer Neugestaltung — gepaart werden muss, die die zugrunde liegende Operation sicher wiederholbar macht.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken, um diese Lösung im Kontext eines Legacy-Systems umzusetzen.

- Identifizieren Sie Operationen im Legacy-System, die aufgrund vorübergehender Probleme fehlschlagen (Netzwerk-Timeouts, temporäre Nichtverfügbarkeit)
- Implementieren Sie Retry-Logik mit exponentiellem Backoff und Jitter, um Thundering-Herd-Probleme zu vermeiden
- Setzen Sie maximale Wiederholungszahlen, um unendliche Schleifen zu verhindern, wenn Fehler anhaltend statt vorübergehend sind
- Klassifizieren Sie Fehler als wiederholbar (Timeout, Verbindung abgelehnt) versus nicht wiederholbar (Authentifizierungsfehler, Validierungsfehler)
- Kombinieren Sie Wiederholungen mit Circuit Breakern, um Wiederholungen zu stoppen, wenn eine Abhängigkeit eindeutig ausgefallen ist
- Stellen Sie sicher, dass Operationen idempotent sind, bevor Sie Retry-Logik hinzufügen, um doppelte Seiteneffekte zu verhindern
- Protokollieren Sie Wiederholungsversuche mit Kontext, um bei der Identifikation chronischer vorübergehender Fehlerquellen zu helfen

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie dafür aufgeben.

**Vorteile:**
- Erholt sich automatisch von vorübergehenden Fehlern ohne manuellen Eingriff
- Verbessert die wahrgenommene Zuverlässigkeit, indem temporäre Infrastrukturprobleme maskiert werden
- Einfach zu implementieren und fügt Legacy-Integrationspunkten Resilienz hinzu
- Reduziert die Häufigkeit nutzersichtbarer Fehler und Support-Tickets

**Kosten und Risiken:**
- Wiederholungen bei nicht idempotenten Operationen können doppelte Daten oder Transaktionen verursachen
- Aggressives Wiederholen ohne Backoff kann Last auf bereits gestressten Systemen verstärken
- Das Wiederholen anhaltend fehlschlagender Operationen verschwendet Ressourcen und verzögert Fehlerberichterstattung
- Das Maskieren vorübergehender Fehler kann systemische Probleme verbergen, die untersucht werden müssen

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Ein Legacy-Bestellverwaltungssystem scheiterte häufig an der Kommunikation mit einer externen Versanddienstleister-API aufgrund kurzer Netzwerkunterbrechungen zwischen Rechenzentren. Jeder Fehlschlag erforderte manuelle erneute Einreichung durch Kundendienstmitarbeiter. Durch das Hinzufügen von Retry-Logik mit exponentiellem Backoff (1 s, 2 s, 4 s) und maximal drei Versuchen erholte sich das System automatisch von über 98 % der vorübergehenden Fehler. Die verbleibenden 2 %, die Wiederholungen erschöpften, wurden automatisch für manuelle Überprüfung eingereiht, was die Kundendienst-Arbeitslast für versandbezogene Probleme um 95 % reduzierte.
