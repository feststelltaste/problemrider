---
title: Konfigurations-Drift
description: Systemkonfigurationen weichen im Laufe der Zeit schrittweise von den
  vorgesehenen Standards ab, was Inkonsistenzen und Zuverlässigkeitsprobleme erzeugt.
category:
- Architecture
- Operations
related_problems:
- slug: configuration-chaos
  similarity: 0.75
- slug: regulatory-compliance-drift
  similarity: 0.65
- slug: inadequate-configuration-management
  similarity: 0.65
- slug: deployment-environment-inconsistencies
  similarity: 0.65
- slug: legacy-configuration-management-chaos
  similarity: 0.65
- slug: change-management-chaos
  similarity: 0.65
solutions:
- infrastructure-as-code
- automated-migration-tools
- backup-and-recovery
- compatibility-matrix
- containerization
- containerized-databases
- dependency-pinning
- environment-parity
- externalized-configuration
- immutable-infrastructure
- isolated-test-environments
- monitoring-system-integrity
- multi-cloud-iac
- platform-independent-configuration-files
- platform-independent-configuration-management
- production-environment-maintenance
- regular-maintenance-and-updates
- restore-points
- secure-by-default
- secure-configuration
- security-audits
- security-monitoring
- standardized-deployment-scripts
- version-control
- virtual-development-environments
- virtual-networks
- virtualization
- certificate-management
- configuration-checks
- environment-variables-for-configuration
- vulnerability-scans
- zero-trust-architecture
- customization-under-version-control
layout: problem
lang: de
en_slug: configuration-drift
---

## Description

Konfigurations-Drift entsteht, wenn sich Systemkonfigurationen im Laufe der Zeit schrittweise von ihrem vorgesehenen oder dokumentierten Zustand entfernen, was zu Inkonsistenzen zwischen Umgebungen, unerwartetem Systemverhalten und verringerter Zuverlässigkeit führt. Diese Drift kann durch manuelle Änderungen, nicht ordentlich kontrollierte automatisierte Prozesse oder die schrittweise Anhäufung von Änderungen entstehen, die nicht nachverfolgt oder standardisiert werden.

## Indicators ⟡

- Produktionssysteme verhalten sich anders als Staging- oder Entwicklungsumgebungen
- Konfigurationsdateien unterscheiden sich über vermeintlich identische Systeme hinweg
- Das Systemverhalten ändert sich unerwartet ohne entsprechende Codeänderungen
- Manuelle Konfigurationsänderungen werden nicht dokumentiert oder nachverfolgt
- Automatisierte Deployments schlagen aufgrund umgebungsspezifischer Konfigurationen fehl

## Symptoms ▲

- [Inkonsistentes Verhalten](inkonsistentes-verhalten.md)
<br/>  Während Konfigurationen von ihrem vorgesehenen Zustand abweichen, liefern dieselben Operationen unterschiedliche Ergebnisse über unterschiedliche Umgebungen oder Instanzen hinweg.
- [Inkonsistenzen zwischen Deployment-Umgebungen](inkonsistenzen-zwischen-deployment-umgebungen.md)
<br/>  Konfigurations-Drift verursacht direkt unterschiedliches Verhalten von Umgebungen, was Deployments unvorhersehbar macht.
- [Debugging-Schwierigkeiten](debugging-schwierigkeiten.md)
<br/>  Wenn Konfigurationen von ihrem dokumentierten Zustand abgedriftet sind, können Entwickler Produktionsprobleme in anderen Umgebungen nicht reproduzieren.
- [Konfigurationschaos](konfigurationschaos.md)
<br/>  Schrittweise Drift in einzelnen Konfigurationen häuft sich zu allgemeinem Konfigurationschaos an, wenn sie über mehrere Systeme hinweg unadressiert bleibt.
- [Unvorhersehbares Systemverhalten](unvorhersehbares-systemverhalten.md)
<br/>  Gedriftete Konfigurationen verursachen unerwartete Nebeneffekte, da der tatsächliche Systemzustand nicht mehr dem entspricht, was Entwickler und Betreiber erwarten.

## Causes ▼

- [Unzureichendes Konfigurationsmanagement](unzureichendes-konfigurationsmanagement.md)
<br/>  Ohne ordentliche Konfigurationsnachverfolgung und Baselines gibt es keinen Mechanismus, um schrittweise Drift zu erkennen oder zu verhindern.
- [Manuelle Deployment-Prozesse](manuelle-deployment-prozesse.md)
<br/>  Manuelle Konfigurationsänderungen sind anfällig für Inkonsistenz und bleiben oft undokumentiert, was direkt dazu führt, dass Konfigurationen im Laufe der Zeit driften.
- [Fehlende Eigenverantwortung und Rechenschaftspflicht](fehlende-eigenverantwortung-und-rechenschaftspflicht.md)
<br/>  Wenn niemand für die Aufrechterhaltung von Konfigurationsstandards verantwortlich ist, häufen sich Ad-hoc-Änderungen ohne Review oder Korrektur an.
- [Informationsverfall](informationsverfall.md)
<br/>  Während Dokumentation über vorgesehene Konfigurationen veraltet, verlieren Teams die Baseline, die nötig ist, um Drift zu erkennen und zu korrigieren.

## Detection Methods ○

- **Konfigurations-Monitoring:** Kontinuierliche Überwachung von Konfigurationsdateien auf Änderungen
- **Umgebungsvergleich:** Regelmäßiger Vergleich von Konfigurationen über unterschiedliche Umgebungen hinweg
- **Konfigurations-Audit:** Periodische Prüfung tatsächlicher Konfigurationen gegen dokumentierte Standards
- **Drift-Erkennungswerkzeuge:** Nutzung von Werkzeugen, die automatisch Konfigurationsänderungen und Drift erkennen
- **Baseline-Konfigurationsmanagement:** Pflege und Vergleich gegen bekannte gute Konfigurations-Baselines

## Examples

Eine Webanwendung läuft in der Entwicklung einwandfrei, schlägt aber in der Produktion intermittierend fehl, aufgrund unterschiedlicher Datenbankverbindungs-Timeout-Einstellungen, die vor Monaten manuell angepasst wurden. Das Produktions-Datenbank-Timeout wurde erhöht, um lang laufende Abfragen zu bewältigen, aber diese Änderung wurde nie dokumentiert oder auf andere Umgebungen angewendet. Wenn Entwickler versuchen, Produktionsprobleme zu reproduzieren, können sie das nicht, weil ihre Entwicklungsumgebung ein anderes Timeout-Verhalten hat. Ein weiteres Beispiel betrifft ein Microservices-Deployment, bei dem sich einzelne Service-Konfigurationen im Laufe der Zeit über unterschiedliche Serverinstanzen hinweg schrittweise auseinanderentwickelt haben. Manche Instanzen haben Debug-Logging aktiviert, andere haben unterschiedliche Speicherlimits, und die SSL-Zertifikatsvalidierung variiert zwischen Servern. Diese Konfigurations-Drift macht es unmöglich, das Systemverhalten vorherzusagen und Probleme wirksam zu beheben.
