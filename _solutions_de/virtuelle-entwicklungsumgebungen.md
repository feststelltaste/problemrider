---
title: Virtuelle Entwicklungsumgebungen
description: Bereitstellung von Entwicklungsumgebungen in virtuellen
  Maschinen oder Containern.
category:
- Operations
- Process
problems:
- inefficient-development-environment
- deployment-environment-inconsistencies
- difficult-developer-onboarding
- inadequate-onboarding
- inconsistent-onboarding-experience
- poor-system-environment
- configuration-drift
- development-disruption
- new-hire-frustration
- tool-limitations
layout: solution
lang: de
en_slug: virtual-development-environments
related_solutions:
- slug: containerized-databases
  similarity: 0.8
- slug: development-environment-optimization
  similarity: 0.8
- slug: containerization
  similarity: 0.75
- slug: environment-parity
  similarity: 0.75
- slug: virtual-networks
  similarity: 0.75
- slug: simulation-environments
  similarity: 0.75
---

## Description

Eine virtuelle Entwicklungsumgebung definiert alles, was ein Entwickler braucht, um eine Anwendung auszuführen und zu modifizieren — Dienste, Datenbanken, Message Broker, Abhängigkeiten, Konfiguration — als Code, typischerweise durch Docker Compose, Vagrant oder Devcontainers, und speichert diese Definition in der Versionskontrolle neben der Anwendung selbst, statt in einer Wiki-Seite oder im Gedächtnis eines Kollegen. Dies zielt direkt auf ein chronisches Problem in Legacy-Systemen: Über Jahre wächst der Satz lokaler Abhängigkeiten, die zum Ausführen der Anwendung erforderlich sind, organisch und undokumentiert, bis die Einarbeitung neuer Entwickler von stillem Wissen, einem dauerhaft veralteten Setup-Leitfaden und Tagen von Trial and Error abhängt, bevor überhaupt eine funktionierende lokale Umgebung existiert. Indem die Umgebung kodifiziert statt in Prosa beschrieben wird, kann dieselbe Umgebung identisch auf jeder Maschine mit einem einzigen Befehl reproduziert werden, was sowohl die Einarbeitungsverzögerung als auch die breitere Klasse von "Läuft auf meiner Maschine"-Diskrepanzen beseitigt, die entstehen, wenn die lokalen Setups von Entwicklern still voneinander und von der Produktion über die Zeit abgewichen sind. Es erlaubt Entwicklern auch, mehrere Projekte mit widersprüchlichen Abhängigkeitsversionen auf derselben Maschine ohne Konflikt zu behalten, da die Umgebung jedes Projekts vollständig isoliert ist. Der Kompromiss ist lokaler Ressourcenverbrauch und die laufende Wartungslast, die Umgebungsdefinition aktuell zu halten, während sich die echten Abhängigkeiten des Legacy-Systems weiterentwickeln, aber für Legacy-Systeme mit genuin komplexen lokalen Setup-Anforderungen ist dies üblicherweise ein kleiner Preis gegen die Einarbeitungskosten, die es ersetzt.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken, um diese Lösung im Kontext eines Legacy-Systems umzusetzen.

- Definieren Sie die Entwicklungsumgebung als Code mit Docker Compose, Vagrant oder Devcontainers
- Beziehen Sie alle erforderlichen Dienste, Datenbanken und Abhängigkeiten in die virtuelle Umgebungsdefinition ein
- Speichern Sie die Umgebungsdefinition im selben Repository wie den Anwendungscode zur Versionierung
- Bieten Sie Skripte oder Makefile-Ziele für gängige Operationen (Start, Stop, Reset, Testdaten befüllen)
- Nutzen Sie Volume-Mounts für Quellcode, sodass Entwickler ihre bevorzugte IDE nutzen können, während die Anwendung im Container läuft
- Dokumentieren Sie, wie die virtuelle Umgebung im Projekt-README eingerichtet und genutzt wird
- Halten Sie die virtuelle Umgebung mit Produktionskonfigurationen abgestimmt, um Umgebungsparitätsprobleme zu minimieren

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie dafür aufgeben.

**Vorteile:**
- Neue Entwickler können eine funktionierende Umgebung in Minuten statt Tagen haben
- Beseitigt "Läuft auf meiner Maschine"-Probleme durch Standardisierung der Entwicklungsumgebung
- Stellt sicher, dass Entwicklungsumgebungen näher an die Produktion angeglichen sind, was Probleme früher erfasst
- Ermöglicht Entwicklern, gleichzeitig an mehreren Projekten mit widersprüchlichen Abhängigkeiten zu arbeiten

**Kosten und Risiken:**
- Container und virtuelle Maschinen verbrauchen erhebliche lokale Ressourcen (CPU, Speicher, Festplatte)
- Komplexe Legacy-Systeme mit vielen Diensten könnten leistungsstarke Entwicklermaschinen erfordern
- Debugging innerhalb von Containern kann umständlicher sein als lokales Debugging
- Umgebungsdefinitionen erfordern laufende Pflege, während sich Abhängigkeiten und Dienste weiterentwickeln
- Die Performance von Volume-gemountetem Quellcode kann auf manchen Plattformen langsam sein

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Ein Fintech-Unternehmen hatte einen Legacy-Monolithen, der Oracle Database, Redis, RabbitMQ und mehrere Microservices lokal für die Entwicklung laufend erforderte. Das Setup für neue Entwickler dauerte drei bis fünf Tage und war in einem 40-seitigen Wiki dokumentiert, das dauerhaft veraltet war. Das Team erstellte eine Docker-Compose-Umgebung mit allen vorkonfigurierten Abhängigkeiten und einem Seed-Skript, das Testdaten lud. Die Einarbeitung sank auf unter zwei Stunden. Als ein Datenbank-Versions-Upgrade nötig war, aktualisierte das Team das Docker-Image-Tag, und jeder Entwickler erhielt die neue Version bei seinem nächsten `docker compose pull`. Dies beseitigte auch vier anhaltende "Läuft auf meiner Maschine"-Fehler, die das Team monatelang geplagt hatten.
