---
title: System-Hardening
description: Verbesserung des Sicherheitszustands von Systemen und
  Komponenten.
category:
- Security
quality_tactics_url: https://qualitytactics.de/en/security/system-hardening/
problems:
- authentication-bypass-vulnerabilities
- authorization-flaws
- sql-injection-vulnerabilities
- cross-site-scripting-vulnerabilities
- buffer-overflow-vulnerabilities
- secret-management-problems
- data-protection-risk
- insecure-data-transmission
- password-security-weaknesses
- session-management-issues
- error-message-information-disclosure
- log-injection-vulnerabilities
- regulatory-compliance-drift
- insufficient-audit-logging
layout: solution
lang: de
en_slug: security-hardening-process
related_solutions:
- slug: configuration-checks
  similarity: 0.8
- slug: authentication
  similarity: 0.75
- slug: patch-management
  similarity: 0.75
- slug: secure-protocols
  similarity: 0.75
- slug: cryptographic-methods
  similarity: 0.75
- slug: secret-management
  similarity: 0.75
---

## Description

System-Hardening schließt unnötige Dienste, Standardanmeldedaten und veraltete Protokollkonfigurationen auf der Infrastrukturebene und hebt die Sicherheitslage eines Servers von dem an, was in seinem ursprünglichen Bereitstellungsjahr akzeptabel war, auf etwas heute Vertretbares. Legacy-Server sind hier überproportional exponiert, weil ihre Konfiguration üblicherweise nie seit der anfänglichen Installation überprüft wurde — vergessene FTP- oder Telnet-Daemons, ein Administrator-Passwort, das niemand in einem Jahrzehnt geändert hat, TLS-Einstellungen, die aktuelle Mindeststandards vordatieren — nichts davon erfordert überhaupt eine Anwendungsschwachstelle, um ausgenutzt zu werden. Da Hardening vollständig auf der Infrastrukturebene operiert, ist es eine der wenigen bedeutenden Sicherheitsverbesserungen, die erreichbar sind, selbst wenn der Quellcode der Legacy-Anwendung nicht angefasst werden kann, obwohl es zuerst sorgfältig in Staging getestet werden muss, da eine Legacy-Anwendung still von genau der freizügigen Standardeinstellung abhängen kann, die eine Hardening-Benchmark zu entfernen empfiehlt.

## How to Apply ◆

> Legacy-Systeme werden oft mit der Sicherheitslage ihres ursprünglichen Veröffentlichungsjahrs bereitgestellt — laufende unnötige Dienste, unveränderte Standardanmeldedaten und TLS-Konfigurationen, die vor einem Jahrzehnt veraltete Protokolle erlauben —, was systematisches Hardening zu einer der Sicherheitsinvestitionen mit der höchsten Rendite macht.

- Beginnen Sie mit einer CIS-Benchmark-Level-1-Bewertung der genutzten Betriebssysteme und Middleware; viele Legacy-Server mit Windows Server 2012 oder älteren Linux-Distributionen werden Dutzende Level-1-Befunde haben, einfach weil Standardeinstellungen aus der Ära ihrer ursprünglichen Installation nie überprüft wurden.
- Scannen Sie alle netzwerkseitigen Ports mit nmap und vergleichen Sie die Ergebnisse mit dem, was jeder Server tatsächlich bedienen soll; Legacy-Systeme haben routinemäßig Dienste, die auf Ports lauschen, von denen niemand im aktuellen Team weiß, übrig geblieben aus Entwicklung, Testing oder früherer Funktionalität.
- Deaktivieren oder deinstallieren Sie jeden Dienst, jedes Protokoll und Paket, das für die Funktion der Anwendung nicht benötigt wird; auf jahrelang laufenden Servern bedeutet dies typischerweise, Compiler, FTP-Daemons, alte SMB-Versionen, Telnet und Debug-Utilities zu entfernen, die sich während der Entwicklung angesammelt haben.
- Rotieren Sie alle Standard- und gemeinsamen Anmeldedaten sofort — Datenbank-Admin-Passwörter, Verwaltungskonsolen von Anwendungsservern, Netzwerkgeräte-Admin-Konten —, wobei Sie jede Änderung durch den Secret-Management-Prozess dokumentieren statt in einer gemeinsamen Tabellenkalkulation.
- Härten Sie TLS-Konfigurationen, indem Sie TLS 1.0 und 1.1 deaktivieren und schwache Cipher Suites entfernen; Legacy-Anwendungen handeln häufig die schwächste von beiden Seiten unterstützte Chiffre aus, und viele wurden bereitgestellt, bevor TLS 1.2 der Mindeststandard war.
- Automatisieren Sie die Hardening-Konfiguration mittels Ansible oder Puppet, sodass der gehärtete Zustand kontinuierlich durchgesetzt wird; jahrelang manuell verwaltete Legacy-Server haben Konfigurationsdrift, die Schwachstellen nach jedem Software-Update oder manuellen Eingriff wieder einführt.
- Wenden Sie das Prinzip der geringsten Rechte auf alle Servicekonten an: Datenbanknutzer, Anwendungsserverprozesse und Batch-Job-Konten sollten nur die Berechtigungen haben, die ihre spezifische Arbeitslast erfordert — nicht DBA- oder lokale Administratorrechte, die vor Jahren aus Bequemlichkeit gewährt wurden.
- Planen Sie erneute Hardening-Validierung nach jedem größeren Software-Update oder Betriebssystem-Upgrade, da Paket-Updates häufig Konfigurationsparameter auf ihre Standardwerte zurücksetzen; automatisiertes Compliance-Scanning mit OpenSCAP oder CIS-CAT erfasst diese Regressionen, bevor Angreifer es tun.

## Tradeoffs ⇄

> System-Hardening reduziert die Angriffsfläche von Legacy-Systemen auf messbare, auditierbare und ohne Neuschreibung von Anwendungscode erreichbare Weise, aber übermäßig aggressives Hardening kann Funktionalität brechen, die von den freizügigen Standardeinstellungen abhängt, um die das Legacy-System herum gebaut wurde.

**Vorteile:**

- Hardening schließt Angriffsvektoren, die keine Anwendungsebenen-Schwachstelle zur Ausnutzung benötigen — Standardanmeldedaten, offene Verwaltungsports und unnötige Dienste waren der Einstiegspunkt für größere Sicherheitsverletzungen gegen Systeme ohne bekannte CVEs.
- CIS-Benchmark-Compliance liefert eine vertretbare, auditierbare Sicherheitsbasislinie, die regulatorische Frameworks (PCI DSS, HIPAA, ISO 27001) erfüllt, ohne Änderungen auf Anwendungsebene an der Legacy-Codebasis zu erfordern.
- Automatisiertes Hardening durch Konfigurationsmanagement-Werkzeuge stellt Konsistenz über Flotten von Legacy-Servern sicher, die zuvor manuell verwaltet wurden und über Jahre auf undokumentierte Weise auseinandergedriftet waren.
- Die Deaktivierung unnötiger Dienste reduziert die betriebliche Angriffsfläche jedes Servers, was bedeutet, dass eine Schwachstelle in einem Dienst, den die Anwendung nicht einmal nutzt, nicht gegen sie ausgenutzt werden kann.
- Hardening ist eine der wenigen Sicherheitsverbesserungen, die auf Legacy-Systeme ohne Quellcode-Zugriff angewendet werden können — es operiert auf der Infrastrukturebene, was es selbst dann erreichbar macht, wenn der Anwendungscode nicht geändert werden kann.

**Kosten und Risiken:**

- Legacy-Anwendungen hängen häufig von Verhaltensweisen ab, die Hardening-Benchmarks zu entfernen empfehlen — spezifische TLS-Versionen, unverschlüsselte Protokolle oder breite Dateisystemberechtigungen —, was sorgfältiges Testen erfordert, bevor Level-1-Empfehlungen ausnahmslos angewendet werden.
- Übermäßig aggressives Hardening ohne vorheriges Testen in einer Staging-Umgebung kann Dienste oder Ports deaktivieren, von denen interne Anwendungskomponenten still abhängen, was Ausfälle verursacht, die ohne Kenntnis dessen, was sich geändert hat, schwer zu diagnostizieren sind.
- Die Aufrechterhaltung gehärteter Konfigurationen erfordert laufenden Aufwand, während Betriebssystem-Updates, Middleware-Upgrades und neue Anwendungsbereitstellungen Einstellungen auf ihre Standardwerte zurücksetzen; Teams ohne Automatisierung werden sehen, wie ihre gehärtete Basislinie innerhalb von Monaten erodiert.
- Ausnahmemanagement für Legacy-Anwendungen, die spezifische Hardening-Empfehlungen nicht erfüllen können, fügt administrativen Overhead hinzu und schafft eine dokumentierte Liste akzeptierter Risiken, die periodisch überprüft und neu gerechtfertigt werden muss.
- Entwicklungs- und Testumgebungen für Legacy-Systeme können typischerweise nicht im selben Maße wie Produktion gehärtet werden, was eine Konfigurationslücke schafft, die bedeutet, dass Probleme erst bei der Bereitstellung erfasst werden — was das Risiko erhöht, dass hardening-induzierte Regressionen Produktion erreichen.

## How It Could Be

> Die folgenden Szenarien veranschaulichen System-Hardening, angewendet in realistischen Legacy-System-Modernisierungsaufwänden.

Das Legacy-MES (Manufacturing Execution System) eines Fertigungsunternehmens lief auf Windows-Server-2008-Servern, die elf Jahre in Produktion gewesen waren. Eine Sicherheitsbewertung offenbarte, dass die Server IIS 6, FTP, Telnet und mehrere andere Dienste liefen, die nach der initialen Installation nie deaktiviert worden waren. Drei der Server hatten noch das Standard-lokale-Administrator-Passwort des Serveranbieters. Nachdem das Team CIS-Benchmark-Level-1-Einstellungen mittels eines Ansible-Playbooks anwendete, 12 unnötige Dienste pro Server deaktivierte und alle Anmeldedaten rotierte, zeigte ein Folge-Schwachstellenscan eine Reduktion von 47 hohen/kritischen Befunden pro Server auf 6 — keiner davon war ohne gültige Anmeldedaten remote ausnutzbar.

Eine Gesundheitsorganisation, die ein Legacy-Patientenportal betrieb, entdeckte während eines PCI-Audits, dass ihre Anwendungsserver TLS 1.0 aushandelten und RC4-Cipher-Suites akzeptierten — beides veraltet und unter aktuellen PCI-DSS-Anforderungen verboten. Die Server waren 2011 bereitgestellt worden, als diese Einstellungen akzeptabel waren, und niemand hatte seither die TLS-Konfiguration überprüft. Die Aktualisierung der Apache-Konfiguration zur Erzwingung von mindestens TLS 1.2 und Beschränkung der Cipher Suites auf die genehmigte Liste erforderte ein zweistündiges Wartungsfenster und verursachte ein kurzes Kompatibilitätsproblem mit einem internen Monitoring-Werkzeug, das noch einen TLS-1.0-Client nutzte — selbst als separater Behebungspunkt identifiziert. Die Organisation bestand den TLS-Teil des Audits ohne jegliche Anwendungscodeänderungen.

Ein Logistikunternehmen, das eine Cloud-Migration seines On-Premises-Legacy-Systems durchführte, nutzte die Migration als Gelegenheit, gehärtete Golden-AMIs basierend auf CIS-Amazon-Linux-2-Benchmark-Empfehlungen zu etablieren. Jede aus diesen Images gestartete neue EC2-Instanz begann mit CIS-Level-1-Compliance, was den manuellen Hardening-Aufwand beseitigte, der inkonsistente Ergebnisse über die On-Premises-Flotte hinweg produziert hatte. AWS-Config-Regeln bewerteten laufende Instanzen kontinuierlich gegen die Benchmark und alarmierten das Operations-Team innerhalb von Minuten, wenn eine Instanz von der gehärteten Basislinie abdriftete — was einen vierteljährlichen manuellen Auditprozess ersetzte, der Lücken erst Monate nach ihrer Entstehung erfasst und geschlossen hatte.
