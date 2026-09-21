---
title: Konfigurationsprüfungen
description: Dokumentation und regelmäßige Überprüfung sicherheitsrelevanter Einstellungen.
category:
- Security
- Operations
problems:
- configuration-drift
- configuration-chaos
- inadequate-configuration-management
- legacy-configuration-management-chaos
- regulatory-compliance-drift
- deployment-environment-inconsistencies
- secret-management-problems
- environment-variable-issues
- logging-configuration-issues
- customization-outside-version-control
layout: solution
lang: de
en_slug: configuration-checks
related_solutions:
- slug: security-audits
  similarity: 0.8
- slug: security-hardening-process
  similarity: 0.8
- slug: secure-configuration
  similarity: 0.8
- slug: vulnerability-scans
  similarity: 0.75
- slug: logging-and-monitoring
  similarity: 0.75
- slug: penetration-tests
  similarity: 0.75
---

## Description

Konfigurationsprüfungen dokumentieren die sicherheitsrelevanten Einstellungen, mit denen ein System laufen soll — TLS-Versionen und Cipher Suites, Firewall-Regeln, Authentifizierungsparameter, Dateiberechtigungen, Service-Account-Privilegien — als explizite Baseline, und verifizieren dann das laufende System gegen diese Baseline in wiederkehrendem Rhythmus statt nur beim initialen Setup. Legacy-Systeme häufen Konfigurations-Drift fast standardmäßig an: Einstellungen werden während Vorfallreaktion geändert und nie zurückgesetzt, Standard-Anmeldeinformationen überleben, weil niemand daran dachte, sie zu rotieren, und Services, die für eine längst vergessene Debugging-Sitzung aktiviert wurden, lauschen Jahre später noch. Nichts davon zeigt sich als Codeänderung, sodass Code-Review und Versionskontrolle — die üblichen Sicherheitsnetze für Legacy-Systeme — es nicht abfangen, was genau der Grund ist, warum ein separater, ständiger Verifikationsmechanismus benötigt wird. Automatisierte Scan-Werkzeuge vergleichen den tatsächlichen Zustand mit der dokumentierten Baseline und bringen Abweichungen als Befunde ans Licht, was unsichtbare Drift in etwas verwandelt, auf das das Team reagieren kann, bevor es stattdessen von einem Angreifer oder Auditor entdeckt wird. Weil die Prüfung nur so gut ist wie die Baseline dahinter, erfordert die Praxis Vorabaufwand, um zu erfassen, was „korrekt konfiguriert" für eine gegebene Legacy-Komponente bedeutet, und laufende Pflege, um diese Definition aktuell zu halten, während legitime Änderungen vorgenommen werden.

## How to Apply ◆

> Legacy-Systeme häufen über Jahre Ad-hoc-Änderungen, Patches und Personalfluktuation Sicherheitsfehlkonfigurationen an. Konfigurationsprüfungen dokumentieren systematisch erwartete Sicherheitseinstellungen und verifizieren sie regelmäßig gegen den tatsächlichen Systemzustand.

- Erstellen Sie eine Sicherheitskonfigurations-Baseline, die alle sicherheitsrelevanten Einstellungen dokumentiert: Firewall-Regeln, TLS-Versionen, Cipher Suites, Authentifizierungsparameter, Logging-Levels, Dateiberechtigungen, Datenbankzugriffskontrollen und Service-Account-Privilegien.
- Implementieren Sie automatisiertes Konfigurationsscanning unter Nutzung von Werkzeugen wie CIS Benchmarks, OpenSCAP oder benutzerdefinierten Skripten, die die tatsächliche Systemkonfiguration mit der dokumentierten Baseline vergleichen und Abweichungen melden.
- Planen Sie regelmäßige Konfigurationsaudits (mindestens monatlich), die automatisierte Scans ausführen und Berichte über Konfigurations-Drift von der Sicherheits-Baseline produzieren. Behandeln Sie Abweichungen als Befunde, die Untersuchung und Behebung erfordern.
- Verifizieren Sie, dass Standard-Anmeldeinformationen für alle Systemkomponenten geändert wurden, einschließlich Datenbanken, Message Brokern, administrativen Konsolen und Monitoring-Werkzeugen. Legacy-Systeme behalten häufig werksseitige Standardpasswörter auf internen Komponenten.
- Prüfen Sie, dass unnötige Services, Ports und Features deaktiviert sind. Legacy-Systeme betreiben oft Services, die während der initialen Einrichtung oder Debugging aktiviert und nie entfernt wurden, was die Angriffsfläche unnötig erweitert.
- Überprüfen Sie Datei- und Verzeichnisberechtigungen, um sicherzustellen, dass Konfigurationsdateien, Log-Dateien und Datenverzeichnisse nur für die Nutzer und Prozesse zugänglich sind, die sie brauchen. Legacy-Systeme nutzen häufig übermäßig freizügige Dateiberechtigungen.
- Integrieren Sie Konfigurationsprüfungen in die Deployment-Pipeline, sodass neue Deployments automatisch gegen die Sicherheits-Baseline verifiziert werden, bevor sie live gehen.

## Tradeoffs ⇄

> Konfigurationsprüfungen bieten systematische Sichtbarkeit in sicherheitsrelevante Einstellungen und fangen Drift ab, bevor sie zu einer Schwachstelle wird, erfordern aber Baseline-Definition und laufende Pflege.

**Vorteile:**

- Erkennt Sicherheitsfehlkonfigurationen, bevor sie ausgenutzt werden können, indem tatsächliche Einstellungen mit einer bekannt-guten Baseline verglichen werden.
- Verhindert Konfigurations-Drift, indem Änderungen identifiziert werden, die ohne Befolgung des Change-Management-Prozesses vorgenommen wurden.
- Unterstützt Compliance-Audits, indem dokumentierte Evidenz geboten wird, dass Sicherheitseinstellungen erforderliche Standards erfüllen.
- Verringert das Risiko menschlichen Fehlers während Systemänderungen, indem automatisch verifiziert wird, dass sicherheitskritische Einstellungen korrekt bleiben.

**Kosten und Risiken:**

- Die Erstellung der initialen Konfigurations-Baseline für ein Legacy-System erfordert erheblichen Aufwand, um alle sicherheitsrelevanten Einstellungen zu identifizieren und zu dokumentieren.
- Automatisierte Scan-Werkzeuge könnten False Positives produzieren, die Untersuchung erfordern und Zeit des Sicherheitsteams verbrauchen.
- Die Baseline muss aktualisiert werden, wann immer legitime Konfigurationsänderungen vorgenommen werden, was laufende Pflegelast schafft.
- Konfigurationsprüfungen decken möglicherweise keine Anwendungsebenen-Sicherheitseinstellungen in benutzerdefiniertem Legacy-Code ab und fokussieren sich primär auf Infrastruktur- und Plattformkonfiguration.

## How It Could Be

> Die folgenden Szenarien veranschaulichen, wie Konfigurationsprüfungen Sicherheitsprobleme in Legacy-Systemen erkennen und verhindern.

Ein Legacy-Webanwendungsserver läuft mit aktiviertem TLS 1.0 und 1.1 neben TLS 1.2, weil die ursprüngliche Konfiguration nie aktualisiert wurde, als diese älteren Protokolle abgekündigt wurden. Ein Schwachstellenscanner entdeckt, dass der Server durch TLS 1.0 für den POODLE-Angriff anfällig ist. Das Team implementiert eine Konfigurations-Baseline, die TLS 1.2 als minimale Protokollversion mit einer definierten Menge starker Cipher Suites festlegt. Automatisierte Konfigurationsprüfungen laufen wöchentlich und vergleichen die tatsächliche TLS-Konfiguration mit dieser Baseline. Als ein Systemadministrator die Webserver-Software nach einer Hardware-Migration neu installiert und die Standardkonfiguration TLS 1.0 wieder aktiviert, markiert die nächste Konfigurationsprüfung die Abweichung innerhalb von 24 Stunden, und die Konfiguration wird korrigiert, bevor das System der Schwachstelle exponiert wird.

Ein Legacy-Datenbankserver läuft seit über fünf Jahren mit dem administrativen Standardpasswort für seine Verwaltungskonsole. Die Konsole ist vom internen Netzwerk aus zugänglich, und jeder mit Netzwerkzugriff kann sich als Datenbankadministrator anmelden. Ein Konfigurationsprüfungs-Audit identifiziert dieses Problem zusammen mit 15 weiteren Standard-Anmeldeinformations-Befunden über die Legacy-Infrastruktur. Das Team rotiert alle Standardpasswörter, implementiert eine vierteljährliche Anmeldeinformations-Rotationsrichtlinie und fügt automatisierte Prüfungen hinzu, die verifizieren, dass keine Komponente mit bekannten Standard-Anmeldeinformationen läuft. Der Konfigurationsprüfungsbericht wird zu einem Standardpunkt in der vierteljährlichen Sicherheitsüberprüfung, was sicherstellt, dass neue Installationen und Upgrades keine Standard-Anmeldeinformationen wieder einführen.
