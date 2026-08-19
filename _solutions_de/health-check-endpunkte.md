---
title: Health-Check-Endpunkte
description: Bereitstellung standardisierter Health-Check-APIs für Load-Balancer-
  und Orchestrator-Monitoring.
category:
- Operations
- Architecture
problems:
- monitoring-gaps
- slow-incident-resolution
- system-outages
- single-points-of-failure
- poor-operational-concept
- service-discovery-failures
- load-balancing-problems
layout: solution
lang: de
en_slug: health-check-endpoints
related_solutions:
- slug: ping
  similarity: 0.75
- slug: heartbeat
  similarity: 0.7
- slug: status-monitoring
  similarity: 0.7
- slug: self-test
  similarity: 0.7
- slug: self-monitoring-and-diagnosis
  similarity: 0.7
- slug: monitoring
  similarity: 0.7
---

## Description

Ein Health-Check-Endpunkt ist eine schlanke, standardisierte HTTP-Schnittstelle, die ein Dienst bereitstellt, damit Load Balancer, Orchestratoren und Monitoring-Werkzeuge seinen Status programmatisch abfragen können, statt ihn indirekt aus Dingen wie einem offenen TCP-Port zu erschließen. Das Muster unterscheidet zwischen Liveness-Prüfungen, die nur beantworten, ob der Prozess läuft, und Readiness-Prüfungen, die verifizieren, dass der Dienst tatsächlich eine Anfrage bearbeiten kann — einschließlich seiner kritischen Abhängigkeiten wie Datenbankkonnektivität oder Verfügbarkeit nachgelagerter Systeme. Diese Unterscheidung ist für Legacy-Dienste besonders wichtig, die häufig in Zustände geraten, in denen der Prozess technisch am Leben, aber funktional feststeckt ist — verklemmt, ohne Datenbankverbindungen, oder wartend auf eine nicht antwortende Abhängigkeit —, ein Zustand, den eine einfache Port-Prüfung nicht erkennen kann, eine gut gestaltete Readiness-Prüfung jedoch schon. Health-Endpunkte in Legacy-Komponenten nachzurüsten gibt der Infrastruktur die Information, die sie braucht, um Verkehr automatisch von ungesunden Instanzen weg zu leiten und Deployments sicher zu sequenzieren — Fähigkeiten, die Legacy-Systemen, die vor der Standardisierung dieses Musters gebaut wurden, oft gänzlich fehlen. Weil Health Checks der Input sind, auf den automatisierte Orchestrierung reagiert, liefert eine Prüfung, die zu wenig meldet (bloße Liveness), falsche Zuversicht, während eine, die zu viel prüft (teure nachgelagerte Aufrufe), riskiert, selbst zu einer Performance-Belastung oder Ursache kaskadierenden Ausfalls zu werden — sodass die Abgrenzung dessen, was jede Prüfung tatsächlich verifiziert, selbst eine zentrale Designentscheidung ist.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken, um diese Lösung im Kontext eines Legacy-Systems umzusetzen.

- Fügen Sie Legacy-Diensten schlanke HTTP-Endpunkte hinzu, die Readiness- und Liveness-Status melden
- Beziehen Sie Abhängigkeitsprüfungen (Datenbankkonnektivität, Verfügbarkeit nachgelagerter Dienste) in Health-Antworten ein
- Standardisieren Sie das Antwortformat über alle Dienste hinweg, damit Monitoring-Werkzeuge sie einheitlich parsen können
- Konfigurieren Sie Load Balancer und Orchestratoren, diese Endpunkte für Routing- und Neustartentscheidungen zu nutzen
- Implementieren Sie flache Prüfungen für Liveness (läuft der Prozess) und tiefe Prüfungen für Readiness (kann er Anfragen bedienen)
- Vermeiden Sie teure Operationen in Health Checks, die selbst die Systemperformance verschlechtern könnten
- Fügen Sie Versionsinformation zu Health-Antworten hinzu, um bei der Deployment-Verifikation zu helfen

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie dafür aufgeben.

**Vorteile:**
- Bietet sofortige Sichtbarkeit auf die Dienstgesundheit ohne manuelle Untersuchung
- Ermöglicht automatisiertes Routing von Verkehr weg von ungesunden Instanzen
- Unterstützt Zero-Downtime-Deployments, indem Readiness signalisiert wird, bevor Verkehr akzeptiert wird
- Standardisiert die Gesundheitsberichterstattung über heterogene Legacy-Komponenten hinweg

**Kosten und Risiken:**
- Health-Endpunkte können veraltet oder irreführend werden, wenn sie keine sinnvollen Bedingungen prüfen
- Tiefe Health Checks, die Abhängigkeiten verifizieren, können kaskadierende Ausfälle erzeugen, wenn eine Abhängigkeit langsam ist
- Das Bereitstellen von Health-Endpunkten ohne Authentifizierung kann interne Systeminformation preisgeben
- Das Hinzufügen von Endpunkten zu Legacy-Anwendungen kann Framework-Modifikationen erfordern

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Ein Medienunternehmen betrieb mehrere Legacy-Java-Dienste hinter einem Load Balancer, der sich ausschließlich auf TCP-Port-Prüfungen verließ. Dienste gerieten häufig in Zustände, in denen der Port offen war, aber die Anwendung verklemmt war oder ihre Datenbankverbindung verloren hatte. Durch das Hinzufügen standardisierter Health-Check-Endpunkte, die Thread-Pool-Verfügbarkeit und Datenbankkonnektivität verifizierten, konnte der Load Balancer ungesunde Instanzen automatisch aus der Rotation entfernen. Dies verringerte nutzersichtbare Fehler um 60 % und gab dem Betriebsteam klare diagnostische Information bei der Untersuchung von Vorfällen.
