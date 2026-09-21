---
title: Schwachstellenscans
description: Regelmäßige Überprüfung von Systemen und Anwendungen auf
  bekannte Schwachstellen.
category:
- Security
- Operations
problems:
- monitoring-gaps
- obsolete-technologies
- configuration-drift
- insufficient-testing
- quality-blind-spots
- regulatory-compliance-drift
- high-defect-rate-in-production
layout: solution
lang: de
en_slug: vulnerability-scans
related_solutions:
- slug: security-audits
  similarity: 0.85
- slug: third-party-dependency-check
  similarity: 0.8
- slug: security-tests
  similarity: 0.8
- slug: static-code-analysis
  similarity: 0.8
- slug: patch-management
  similarity: 0.8
- slug: monitoring-system-integrity
  similarity: 0.8
---

## Description

Ein Schwachstellenscan ist eine automatisierte, wiederholbare Prüfung von Infrastruktur- und Anwendungskomponenten gegen Datenbanken bekannter Sicherheitsschwächen, ausgeführt nach einem definierten Zeitplan, sodass Exposition proaktiv entdeckt wird, statt erst, nachdem ein Vorfall oder ein Compliance-Audit die Frage erzwingt. Scan-Werkzeuge decken sowohl die Infrastrukturschicht — Betriebssysteme, Middleware, Anwendungsserver — als auch die Anwendungsschicht ab, und die Praxis hängt von einem Workflow ab, der Befunde in Issue-Tracking mit schweregradbasierten Sanierungsfristen leitet, gefolgt von erneutem Scannen, um zu bestätigen, dass der Fix die Lücke tatsächlich geschlossen hat. Für Legacy-Systeme ist regelmäßiges Scannen oft die erste systematische Sicherheitsaktivität, die ein Team unternimmt, weil der Technologie-Stack üblicherweise Jahre ungepatchter Komponenten und Konfigurationsdrift angesammelt hat, ohne dass jemand ein aktuelles, vollständiges Bild dessen pflegt, was tatsächlich exponiert ist — der Scan führt effektiv das Asset-Inventar und die Risikobewertung durch, mit denen manuelle Nachverfolgung nicht Schritt halten konnte. Hier zählen auch die Grenzen des Scannens am meisten in einem Legacy-Kontext: Befunde übersteigen oft das, was angesichts der technologischen Einschränkungen alter Plattformen schnell saniert werden kann, falsch positive Ergebnisse erfordern Experten-Triage, die mit anderer Wartungsarbeit konkurriert, und aggressive Scans können gelegentlich brüchige Legacy-Komponenten destabilisieren, die nie gegen unerwartete Traffic-Muster gehärtet wurden. Trotz dieser Kosten gibt die resultierende Trenddaten dem Management eine konkrete, sich verbessernde Baseline für die Sicherheitslage und bringt häufig vergessene Assets zutage — End-of-Life-Systeme, an deren fortgesetzten Betrieb sich niemand erinnerte —, die sonst unsichtbar blieben, bis sie ein Problem verursachten.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken, um diese Lösung im Kontext eines Legacy-Systems umzusetzen.

- Setzen Sie automatisierte Schwachstellen-Scanning-Werkzeuge sowohl für Infrastruktur (z. B. Nessus, OpenVAS) als auch für Anwendungen (z. B. OWASP ZAP) ein
- Planen Sie regelmäßige Scans mit einer Häufigkeit, die dem Risikoniveau und der Änderungsrate des Systems angemessen ist
- Scannen Sie alle Komponenten einschließlich Betriebssysteme, Middleware, Anwendungsserver und maßgeschneiderten Anwendungscode
- Etablieren Sie einen Schwachstellenmanagement-Workflow mit definierten SLAs für Sanierung basierend auf Schweregrad
- Integrieren Sie Scan-Ergebnisse mit Issue-Tracking-Systemen, um sicherzustellen, dass Befunde zugewiesen und nachverfolgt werden
- Validieren Sie Sanierung durch erneutes Scannen nach Anwendung von Fixes
- Generieren Sie regelmäßige Schwachstellen-Trendberichte für Management-Sichtbarkeit

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie dafür aufgeben.

**Vorteile:**
- Bietet systematische Entdeckung bekannter Schwachstellen über die gesamte Systemlandschaft
- Ermöglicht risikobasierte Priorisierung von Patching- und Sanierungsanstrengungen
- Schafft eine kontinuierliche Baseline zur Messung der Verbesserung der Sicherheitslage
- Erfüllt Compliance-Anforderungen für regelmäßige Schwachstellenbewertung

**Kosten und Risiken:**
- Scans können die Systemperformance beeinträchtigen und sollten für Legacy-Systeme während verkehrsarmer Zeiten geplant werden
- Legacy-Systeme könnten viele Befunde produzieren, die aufgrund technologischer Einschränkungen nicht leicht saniert werden können
- Falsch positive Ergebnisse erfordern Experten-Triage und können Sanierungsaufwand verschwenden
- Schwachstellenscanner erkennen nur bekannte Schwachstellen und können keine Zero-Day- oder Logikfehler finden
- Aggressives Scannen kann gelegentlich brüchige Legacy-Komponenten zum Absturz bringen

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Ein Fertigungsunternehmen begann monatliche Schwachstellenscans seiner Legacy-Produktionssteuerungssysteme, nachdem ein Compliance-Audit das Fehlen jeglichen Schwachstellenmanagementprogramms bemängelt hatte. Der erste Scan offenbarte 156 Schwachstellen über 12 Server hinweg, einschließlich 8 kritischer Befunde im Zusammenhang mit ungepatchten Betriebssystemen und exponierten Management-Schnittstellen. Das Team etablierte eine Sanierungstaktung, die kritische Befunde innerhalb von drei Monaten auf null reduzierte und mittlere Befunde innerhalb von sechs Monaten um 70 % senkte. Das Scanning-Programm identifizierte auch drei Server mit End-of-Life-Betriebssystemen, die im Asset-Inventar übersehen worden waren, was eine geplante Migration auslöste.
