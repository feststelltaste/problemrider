---
title: Containerisierte Datenbanken
description: Betrieb von Datenbanken in Containern.
category:
- Database
- Operations
problems:
- deployment-environment-inconsistencies
- inadequate-test-infrastructure
- configuration-drift
- complex-deployment-process
- inefficient-development-environment
- difficult-developer-onboarding
- inadequate-test-data-management
layout: solution
lang: de
en_slug: containerized-databases
related_solutions:
- slug: containerization
  similarity: 0.85
- slug: virtual-development-environments
  similarity: 0.8
- slug: nosql-databases
  similarity: 0.75
- slug: database-abstraction
  similarity: 0.75
- slug: platform-independent-data-storage
  similarity: 0.75
- slug: data-replication
  similarity: 0.75
---

## Description

Containerisierte Datenbanken packen eine Datenbank-Engine, ihre Konfiguration und optional ihr Schema und Seed-Daten in ein Container-Image, sodass eine vollständig funktionsfähige, wegwerfbare Datenbankinstanz auf Anfrage gestartet werden kann, statt auf einen einzelnen gemeinsam genutzten Server angewiesen zu sein. Legacy-Entwicklungseinrichtungen leiten üblicherweise jeden Entwickler und jeden CI-Lauf durch eine gemeinsam genutzte Datenbankinstanz, die in einen inkonsistenten Schemazustand abdriftet, während verschiedene Branches widersprüchliche Migrationen anwenden, und die Testdatenverschmutzung und Bereitstellungsverzögerungen zu einer routinemäßigen Reibungsquelle macht. Jedem Entwickler und jedem CI-Job seine eigene containerisierte, wegwerfbare Instanz zu geben entfernt diese Konkurrenz vollständig: Schema-Migrationen können lokal ausprobiert, gebrochen und zurückgesetzt werden, ohne auf einen DBA zu warten oder sich mit anderen Entwicklern zu koordinieren, und eine frische, isolierte Datenbank ist für jeden Testlauf verfügbar. Dies macht containerisierte Datenbanken besonders wertvoll, um Schema-Migrationen sicher zu validieren, bevor sie eine gemeinsam genutzte Umgebung berühren, da Fehler nur einen wegwerfbaren Container betreffen. Der Ansatz eignet sich am besten für Entwicklung, Testing und CI statt Produktionsnutzung wie sie ist, weil er nicht automatisch produktionsgerechte Performance-, Backup- und Failover-Eigenschaften repliziert, und sehr große Legacy-Datensätze müssen möglicherweise verkleinert werden, bevor sie in einen praktikablen containerbasierten Workflow passen.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken zur Implementierung dieser Lösung im Kontext eines Legacy-Systems.

- Nutzen Sie containerisierte Datenbanken für Entwicklungs- und Testumgebungen, um Konsistenz mit Produktionsschemata sicherzustellen
- Erstellen Sie Datenbank-Container-Images, vorgeladen mit Schema-Migrationen und Seed-Daten, für schnelle Umgebungsbereitstellung
- Nutzen Sie Docker-Volumes für persistenten Speicher, sodass der Datenbankzustand Container-Neustarts während der Entwicklung überlebt
- Konfigurieren Sie Health Checks, die verifizieren, dass die Datenbank bereit ist, bevor abhängige Services starten
- Nutzen Sie Docker Compose, um die Datenbank neben der Anwendung für lokale Entwicklung zu orchestrieren
- Bewerten Sie für Produktion verwaltete Datenbankservices gegenüber selbst verwalteten containerisierten Datenbanken basierend auf operativer Reife
- Automatisieren Sie die Bereitstellung von Datenbank-Containern in CI/CD-Pipelines, sodass jeder Testlauf eine frische, isolierte Datenbankinstanz erhält

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie aufgeben.

**Vorteile:**
- Ermöglicht jedem Entwickler, eine isolierte Datenbankinstanz zu betreiben, die der Produktionskonfiguration entspricht
- Eliminiert Konflikte gemeinsam genutzter Entwicklungsdatenbanken und Testdatenverschmutzung
- Macht Datenbankbereitstellung für CI/CD-Pipelines schnell und wiederholbar
- Vereinfacht das Testen von Datenbankmigrationen durch bedarfsgesteuertes Hochfahren frischer Instanzen

**Kosten und Risiken:**
- Containerisierte Datenbanken könnten Produktions-Performance-Eigenschaften nicht perfekt replizieren
- Die Verwaltung persistenten Speichers in Containern erfordert sorgfältige Volume-Konfiguration
- Die Produktionsnutzung containerisierter Datenbanken erfordert Expertise in Storage-Treibern, Backup-Strategien und Failover
- Große Legacy-Datenbanken könnten für Entwicklung unpraktikabel zu containerisieren sein, wenn der Datensatz nicht sinnvoll verkleinert werden kann
- Datenbank-Lizenzbedingungen könnten Container-Deployment einschränken oder komplizieren

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Ein Entwicklungsteam teilte sich eine einzelne Oracle-Entwicklungsdatenbank, die häufig inkonsistente Schemazustände hatte, was Testfehler verursachte und Entwickler blockierte. Das Team erstellte ein PostgreSQL-Container-Image, vorgeladen mit dem migrierten Schema und repräsentativen Seed-Daten. Jeder Entwickler betrieb seine eigene Datenbankinstanz lokal, und CI-Pipelines fuhren frische Container für jeden Testlauf hoch. Schema-Migrationstests wurden trivial: Entwickler wendeten Migrationen auf ihren lokalen Container an und verifizierten Ergebnisse sofort, statt auf einen DBA zu warten, der die gemeinsame Instanz aktualisiert. Die isolierten Umgebungen eliminierten Interferenz zwischen Entwicklern und verringerten datenbankbezogene Build-Fehler um 90 Prozent.
