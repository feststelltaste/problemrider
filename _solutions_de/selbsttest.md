---
title: Selbsttest
description: Fähigkeit einer Komponente, ihren eigenen Zustand und ihre
  Funktionsfähigkeit zu prüfen.
category:
- Testing
- Operations
problems:
- monitoring-gaps
- unpredictable-system-behavior
- slow-incident-resolution
- inadequate-integration-tests
- deployment-risk
- system-outages
- dma-coherency-issues
layout: solution
lang: de
en_slug: self-test
related_solutions:
- slug: self-monitoring-and-diagnosis
  similarity: 0.8
- slug: ping
  similarity: 0.7
- slug: smoke-testing
  similarity: 0.7
- slug: health-check-endpoints
  similarity: 0.7
- slug: automated-tests
  similarity: 0.7
- slug: status-monitoring
  similarity: 0.65
---

## Description

Ein Selbsttest ist eine Prüfung, die eine Komponente beim Start oder periodisch während des Betriebs gegen ihre eigenen Abhängigkeiten und Konfiguration durchführt — Datenbankverbindung, erforderliche Umgebungsvariablen, Erreichbarkeit von Drittanbieter-APIs —, wobei sie sich weigert, Traffic anzunehmen, bis jede Prüfung besteht. Legacy-Deployments neigen besonders dazu, genau auf die Weisen zu versagen, die Selbsttests erkennen sollen: eine fehlende Umgebungsvariable oder ein veralteter Verbindungsstring, der sonst erst auftreten würde, wenn echter Traffic auf den defekten Codepfad trifft, wonach jemand Zeit damit verbringen muss, manuell zur tatsächlichen Ursache zurückzuverfolgen. Diese Prüfungen automatisch beim Start durchzuführen, verwandelt einen langsamen, manuellen Diagnoseprozess in eine unmittelbare, spezifische Fehlermeldung, und die Nutzung von Selbsttest-Ergebnissen als Deployment-Gate verhindert, dass eine ungesunde Instanz überhaupt jemals Produktions-Traffic erhält — obwohl die Tests schnell und frei von Nebenwirkungen bleiben müssen, da ein langsamer oder übermäßig invasiver Selbsttest seine eigenen betrieblichen Probleme schafft.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken, um diese Lösung im Kontext eines Legacy-Systems umzusetzen.

- Implementieren Sie Selbsttests beim Start, die kritische Abhängigkeiten, Konfigurationen und Datenzugriff verifizieren, bevor Traffic angenommen wird
- Fügen Sie periodische Selbsttests hinzu, die während des Betriebs laufen, um Drift oder Degradation von Abhängigkeiten zu erkennen
- Beziehen Sie End-to-End-Smoke-Tests ein, die kritische Geschäftspfade mit synthetischen Transaktionen ausüben
- Gestalten Sie Selbsttests so, dass sie klare Bestanden/Fehlgeschlagen-Ergebnisse mit Diagnoseinformationen bei Fehlschlag produzieren
- Integrieren Sie Selbsttest-Ergebnisse mit Health-Check-Endpunkten und Monitoring-Systemen
- Nutzen Sie Selbsttests als Deployment-Validierungs-Gates, die verhindern, dass ungesunde Instanzen Traffic erhalten
- Halten Sie Selbsttests schnell und leichtgewichtig, um Beeinträchtigungen der Systemperformance oder Startzeit zu vermeiden

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie dafür aufgeben.

**Vorteile:**
- Erfasst Konfigurationsfehler und Abhängigkeitsprobleme sofort beim Start
- Bietet kontinuierliche Validierung, dass das System seine Kernfunktionen ausführen kann
- Reduziert die Zeit für die Diagnose von Problemen, die Selbsttests automatisch identifizieren können
- Verbessert das Deployment-Vertrauen durch Validierung jeder Instanz, bevor sie Traffic bedient

**Kosten und Risiken:**
- Selbsttests, die mit externen Systemen interagieren, können Nebenwirkungen oder Last verursachen
- Langsame Selbsttests verzögern den Start und können schnelle Skalierung stören
- Selbsttest-Pflege fügt laufenden Aufwand hinzu, während sich das System weiterentwickelt
- Falsch positive Selbsttest-Fehlschläge können gesunde Instanzen am Starten hindern

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Ein Legacy-CRM-System schlug häufig nach dem Deployment fehl, wegen fehlender Umgebungsvariablen, falscher Datenbankverbindungsstrings oder nicht verfügbarer Drittanbieterdienste. Ingenieure verbrachten jeweils 30 Minuten mit der manuellen Diagnose jedes Fehlschlags. Durch das Hinzufügen von Selbsttests beim Start, die Datenbankverbindung verifizierten, erforderliche Umgebungsvariablen prüften, API-Schlüssel gegen externe Dienste validierten und eine synthetische Kundensuche durchführten, erkannte das System Konfigurationsprobleme innerhalb von Sekunden nach dem Start und weigerte sich, Traffic anzunehmen, bis alle Prüfungen bestanden. Deployment-Fehlschläge, die zuvor manuelle Untersuchung erforderten, wurden nun sofort durch die Selbsttest-Ausgabe diagnostiziert.
