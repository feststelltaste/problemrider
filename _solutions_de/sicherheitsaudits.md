---
title: Sicherheitsaudits
description: Regelmäßige Überprüfung von Systemen und Prozessen auf
  Sicherheit.
category:
- Security
problems:
- regulatory-compliance-drift
- monitoring-gaps
- quality-blind-spots
- insufficient-audit-logging
- configuration-drift
- data-protection-risk
- secret-management-problems
- authorization-role-explosion
layout: solution
lang: de
en_slug: security-audits
related_solutions:
- slug: vulnerability-scans
  similarity: 0.85
- slug: configuration-checks
  similarity: 0.8
- slug: security-metrics
  similarity: 0.8
- slug: regression-tests
  similarity: 0.8
- slug: security-monitoring
  similarity: 0.75
- slug: security-tests-by-external-parties
  similarity: 0.75
---

## Description

Sicherheitsaudits sind periodische, systematische Überprüfungen des Codes, der Infrastruktur, Konfiguration und Prozesse eines Systems gegen definierte Sicherheitsbasislinien, kombinierend automatisiertes Scanning mit manuellem Expertenurteil, um Drift, Lücken und Verstöße zu erkennen, bevor sie stattdessen durch einen Vorfall entdeckt werden. Der Mechanismus dreht sich grundlegend um die Schaffung eines geplanten Kontrollpunkts: Statt anzunehmen, dass Sicherheitskontrollen, einmal eingerichtet, weiter funktionieren und unbegrenzt korrekt konfiguriert bleiben, verifiziert ein Audit periodisch diese Annahme gegen die Realität, deckt Zugangsrechte, Patch-Stände, Logging-Konfiguration, Verschlüsselung und Drittanbieter-Abhängigkeiten ab. Legacy-Systeme sind ein natürliches Ziel für diese Praxis, weil sie genau die Art unbemerkter Drift ansammeln, die Audits erfassen sollen — Konten, die Mitarbeitern gehören, die vor Jahren gegangen sind, Logging, das während eines vergangenen Wartungsfensters still deaktiviert wurde, Datenbankserver, die ungepatchte Versionen laufen lassen, die niemand markiert hat — Probleme, die schrittweise und ohne auslösendes Ereignis entstehen, sodass nichts an normalem Betrieb sie offenlegen würde. Da Audits Momentaufnahmen sind, können sie kontinuierliche Überwachung nicht ersetzen, aber sie sind gut geeignet, den angesammelten, langsam fortschreitenden Verfall zu erfassen, der für langlebige Systeme mit unvollständiger Dokumentation und verstreuter Eigentümerschaft charakteristisch ist. In einem Legacy-Modernisierungskontext schafft ein regelmäßiger Audit-Rhythmus außerdem die Verantwortlichkeitsstruktur — zugewiesene Befunde, Behebungsfristen und Folgeverifikation —, die nötig ist, um einen Rückstand geerbter Sicherheitsschulden systematisch statt ad hoc abzuarbeiten.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken, um diese Lösung im Kontext eines Legacy-Systems umzusetzen.

- Etablieren Sie einen regelmäßigen Audit-Zeitplan mit definiertem Umfang, der Code, Infrastruktur, Konfigurationen und Prozesse abdeckt
- Nutzen Sie eine Kombination aus automatisierten Scanning-Werkzeugen und manuellem Expertenreview für umfassende Abdeckung
- Prüfen Sie Zugangskontrollen, Logging, Verschlüsselung, Patch-Stände und Konfiguration gegen Sicherheitsbasislinien
- Beziehen Sie Drittanbieter-Abhängigkeiten und Anbieterintegrationen in den Audit-Umfang ein
- Verfolgen Sie Befunde in einem zentralisierten System mit zugewiesenen Eigentümern, Prioritäten und Behebungsfristen
- Führen Sie Folge-Audits durch, um zu verifizieren, dass Befunde ordentlich behoben wurden
- Teilen Sie anonymisierte Audit-Befunde über Teams hinweg, um organisatorisches Lernen zu fördern

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie dafür aufgeben.

**Vorteile:**
- Bietet periodische Zusicherung, dass Sicherheitskontrollen wie beabsichtigt funktionieren
- Identifiziert Drift von Sicherheitsbasislinien, bevor sie zu Vorfällen führt
- Erfüllt regulatorische und Compliance-Anforderungen für Sicherheitsverifikation
- Schafft Verantwortlichkeit durch dokumentierte Befunde und Behebungsverfolgung

**Kosten und Risiken:**
- Audits sind Momentaufnahmen und könnten zwischen Audit-Zyklen eingeführte Probleme übersehen
- Externe Audits können teuer sein, besonders für komplexe Legacy-Umgebungen
- Audit-Müdigkeit kann Teams dazu bringen, Befunde als routinemäßig statt umsetzbar zu behandeln
- Legacy-Systeme mit schlechter Dokumentation machen Audits zeitaufwendiger und weniger gründlich

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Ein Regierungsauftragnehmer, der sein erstes umfassendes Sicherheitsaudit eines 15 Jahre alten Fallverwaltungssystems durchführte, entdeckte, dass 23 Nutzerkonten Mitarbeitern gehörten, die die Organisation vor Jahren verlassen hatten, drei Datenbankserver ungepatchte Versionen mit bekannten kritischen Schwachstellen liefen ließen und Audit-Logging während eines Wartungsfensters zwei Jahre zuvor still deaktiviert worden war. Die Etablierung vierteljährlicher Audits mit automatisierten Vorabprüfungen reduzierte die Anzahl der Befunde pro Audit innerhalb eines Jahres um 60 %, während das Team den Rückstand systematisch adressierte und Wiederauftreten verhinderte.
