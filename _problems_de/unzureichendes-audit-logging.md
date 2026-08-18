---
title: Unzureichendes Audit-Logging
description: Unzureichende Protokollierung sicherheitsrelevanter Ereignisse erschwert
  das Erkennen von Sicherheitsverletzungen, die Untersuchung von Vorfällen oder die
  Wahrung der Compliance.
category:
- Code
- Security
related_problems:
- slug: logging-configuration-issues
  similarity: 0.75
- slug: log-injection-vulnerabilities
  similarity: 0.55
- slug: monitoring-gaps
  similarity: 0.55
- slug: secret-management-problems
  similarity: 0.55
- slug: inadequate-error-handling
  similarity: 0.55
- slug: session-management-issues
  similarity: 0.55
solutions:
- observability-and-monitoring
- security-hardening-process
- audit-trail-management
- authorization-concept
- privacy-by-design
- security-audits
- security-metrics
- security-monitoring
- timestamping
- datensparsamkeit
- digital-forensics
- domain-data-versioning
- honeypots
- logging-and-monitoring
layout: problem
lang: de
en_slug: insufficient-audit-logging
---

## Description

Unzureichendes Audit-Logging tritt auf, wenn Anwendungen es versäumen, sicherheitsrelevante Ereignisse wie Authentifizierungsversuche, Autorisierungsfehler, Datenzugriff, Konfigurationsänderungen oder administrative Aktionen ordentlich zu protokollieren. Dieser Mangel an umfassenden Audit-Trails erschwert es, Sicherheitsverletzungen zu erkennen, Vorfälle zu untersuchen, regulatorische Compliance zu wahren oder Rechenschaftspflicht für Systemaktionen herzustellen.

## Indicators ⟡

- Sicherheitsvorfälle können nicht durch Log-Analyse nachverfolgt werden
- Regulatorische Compliance-Audits scheitern aufgrund fehlender Log-Daten
- Es kann nicht bestimmt werden, wer bestimmte administrative Aktionen durchgeführt hat
- Authentifizierungs- und Autorisierungsereignisse werden nicht protokolliert
- Datenzugriffs- und -modifikationsereignisse werden nicht nachverfolgt

## Symptoms ▲

- [Langsame Vorfallslösung](langsame-vorfallsloesung.md)
<br/>  Ohne umfassende Audit-Logs dauert die Untersuchung und Lösung von Sicherheitsvorfällen viel länger.
- [Regulatorische Compliance-Drift](regulatorische-compliance-drift.md)
<br/>  Fehlende Audit-Trails verursachen Fehlschläge bei Compliance-Audits für Vorschriften wie HIPAA und SOX.
- [Monitoring-Lücken](monitoring-luecken.md)
<br/>  Unzureichendes Logging schafft direkt blinde Flecken im System-Monitoring und in der Observability.
- [Debugging-Schwierigkeiten](debugging-schwierigkeiten.md)
<br/>  Fehlende detaillierte Ereignisprotokolle erschweren die Rückverfolgung und Diagnose von Problemen in Produktion.
- [Datenschutzrisiko](datenschutzrisiko.md)
<br/>  Ohne Audit-Logs kann unbefugter Datenzugriff nicht erkannt oder untersucht werden, was das Datenschutzrisiko erhöht.

## Causes ▼

- [Fehlende Eigenverantwortung und Rechenschaftspflicht](fehlende-eigenverantwortung-und-rechenschaftspflicht.md)
<br/>  Wenn niemand die Logging-Infrastruktur besitzt, werden Audit-Logging-Anforderungen vernachlässigt.
- [Termindruck](termindruck.md)
<br/>  Unter Zeitdruck überspringen Entwickler die Implementierung umfassenden Audit-Loggings, weil es kein sichtbares Feature ist.
- [Unzureichende Design-Fähigkeiten](unzureichende-design-faehigkeiten.md)
<br/>  Entwickler ohne Sicherheitsdesign-Erfahrung verstehen möglicherweise nicht, welche Ereignisse Audit-Logging erfordern.

## Detection Methods ○

- **Analyse der Sicherheitsereignis-Abdeckung:** Überprüfung, welche Sicherheitsereignisse aktuell protokolliert werden
- **Zuordnung von Compliance-Anforderungen:** Zuordnung von Compliance-Anforderungen zu aktuellen Logging-Fähigkeiten
- **Test der Vorfalluntersuchung:** Testen der Fähigkeit, Sicherheitsvorfälle mit verfügbaren Logs zu untersuchen
- **Überprüfung der Vollständigkeit von Audit-Trails:** Verifikation, dass vollständige Audit-Trails für kritische Operationen existieren
- **Bewertung der Nutzeraktivitätsverfolgung:** Bewertung der Abdeckung der Protokollierung von Nutzeraktivität

## Examples

Eine Gesundheitsanwendung verarbeitet Patientenakten, protokolliert aber nur erfolgreiche Datenbankabfragen, nicht fehlgeschlagene Zugriffsversuche oder unbefugte Datenzugriffsversuche. Wenn eine Datenschutzverletzungsuntersuchung stattfindet, können Ermittler nicht bestimmen, welche Konten versuchten, auf bestimmte Patientenakten zuzugreifen, wann unbefugte Zugriffsversuche stattfanden oder den vollen Umfang potenziell kompromittierter Daten nachverfolgen. Das Fehlen umfassenden Audit-Loggings macht es unmöglich, HIPAA-Audit-Anforderungen zu erfüllen oder die Verletzung ordentlich zu untersuchen. Ein weiteres Beispiel betrifft eine Finanzanwendung, die Nutzer-Logins protokolliert, aber nicht, auf welche Daten Nutzer nach der Authentifizierung zugreifen oder diese modifizieren. Wenn verdächtige Aktivität in Kundenkonten entdeckt wird, können Ermittler sehen, wann Nutzer sich anmeldeten, aber nicht bestimmen, welche Finanzdaten eingesehen, modifiziert oder exportiert wurden, was es unmöglich macht, die Auswirkung potenziellen Betrugs oder Datendiebstahls zu bewerten.
