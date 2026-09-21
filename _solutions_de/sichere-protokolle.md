---
title: Sichere Protokolle
description: Nutzung ausschließlich sicherer und aktueller Versionen von
  Netzwerkprotokollen.
category:
- Security
- Operations
problems:
- insecure-data-transmission
- obsolete-technologies
- regulatory-compliance-drift
- data-protection-risk
- poor-system-environment
- technology-lock-in
layout: solution
lang: de
en_slug: secure-protocols
related_solutions:
- slug: cryptographic-methods
  similarity: 0.85
- slug: secure-by-default
  similarity: 0.8
- slug: secure-software
  similarity: 0.8
- slug: encryption
  similarity: 0.75
- slug: vulnerability-scans
  similarity: 0.75
- slug: security-hardening-process
  similarity: 0.75
---

## Description

Sichere Protokolle bedeutet, jegliche Netzwerkkommunikation — Client-Server-Traffic, Service-zu-Service-Aufrufe und administrativen Zugang — auf Protokollversionen und Cipher Suites zu beschränken, die derzeit als kryptografisch solide gelten, und alles andere auszumustern. In der Praxis deckt dies Transportschichtprotokolle wie TLS, Fernzugriffsprotokolle wie SSH und Anwendungsschichtprotokolle wie SMTP oder Datenbank-Wire-Protokolle ab, die alle über die Lebensdauer eines Systems veraltete Versionen ansammeln, während neue Schwachstellen gefunden und alte nur durch Versionsersatz statt an Ort und Stelle gepatcht werden. Legacy-Systeme neigen besonders dazu, veraltete Protokollversionen zu betreiben, weil deren Aktualisierung nie jemandes zugewiesene Verantwortung war, externe Integrationspartner sich auf jener Version festlegten, die existierte, als die Verbindung erstmals gebaut wurde, und das betriebliche Risiko, eine funktionierende Netzwerkkonfiguration anzufassen, proaktive Upgrades entmutigte. Der Mechanismus ist vergleichsweise einfach — Protokollversionsaushandlung ist eine Konfigurationseinstellung auf Servern und Clients statt einer Anwendungscodeänderung —, aber sein Effekt ist unverhältnismäßig: die Beseitigung einer gesamten Klasse bekannter kryptografischer Schwächen auf einmal, statt individuelle Exploits zu patchen, während sie auftauchen. Da Protokolldurchsetzung unterhalb der Anwendungsschicht sitzt, kann sie oft unabhängig von einem breiteren Modernisierungsaufwand ausgerollt werden, was sie zu einer der handhabbareren Sicherheitsverbesserungen für einen Legacy-Bestand macht. Die Hauptbeschränkung in Legacy-Kontexten ist Kompatibilität: alternde Clients, eingebettete Geräte oder Drittanbieter-Integrationen könnten schlicht nicht in der Lage sein, eine aktuelle Protokollversion auszuhandeln, was eine Konfigurationsänderung in ein Koordinations- und Migrationsproblem verwandelt.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken, um diese Lösung im Kontext eines Legacy-Systems umzusetzen.

- Inventarisieren Sie alle Netzwerkprotokolle und ihre vom Legacy-System genutzten Versionen, einschließlich TLS, SSH, SMTP und Datenbankprotokolle
- Deaktivieren Sie veraltete Protokolle wie SSLv3, TLS 1.0 und TLS 1.1 über alle Systemkomponenten hinweg
- Konfigurieren Sie Server und Clients so, dass sie nur aktuelle, sichere Protokollversionen mit starken Cipher Suites nutzen
- Aktualisieren Sie Legacy-Integrationen, die auf veralteten Protokollen beruhen, und bieten Sie Migrationspfade für Drittanbieter-Partner
- Implementieren Sie automatisiertes Scanning, um jegliche unsichere Protokollnutzung im Netzwerk zu erkennen
- Planen und führen Sie Zertifikatsrotationsverfahren für alle TLS-Endpunkte durch

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie dafür aufgeben.

**Vorteile:**
- Schützt Daten während der Übertragung vor Abhören und Manipulation
- Erfüllt Compliance-Anforderungen, die aktuelle Protokollversionen vorschreiben
- Reduziert die Angriffsfläche durch Beseitigung bekannter Protokollschwachstellen
- Verbessert die gesamte Sicherheitslage mit minimalen Anwendungscodeänderungen

**Kosten und Risiken:**
- Legacy-Clients oder Integrationen unterstützen möglicherweise keine modernen Protokollversionen, was koordinierte Upgrades erfordert
- Protokoll-Upgrades können Dienstunterbrechungen verursachen, wenn nicht gründlich getestet
- Manche Legacy-Hardware oder eingebettete Geräte könnten unfähig sein, aktuelle Protokolle zu unterstützen
- Cipher-Suite-Konfiguration erfordert Expertise, um sowohl unsichere als auch inkompatible Wahlen zu vermeiden

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Ein Logistikunternehmen entdeckte während eines Compliance-Audits, dass seine Legacy-Versandintegration noch TLS 1.0 zur Kommunikation mit Transportunternehmen-APIs nutzte. Mehrere Transportunternehmen hatten bereits begonnen, TLS-1.0-Verbindungen abzulehnen, was intermittierende Fehler bei der Versandlabel-Generierung verursachte. Das Team rüstete alle ausgehenden Verbindungen auf TLS 1.2 auf, aktualisierte die interne Zertifizierungsstelle und implementierte ein Protokollversions-Überwachungs-Dashboard. Das Upgrade löste sowohl den Compliance-Befund als auch die intermittierenden Fehler, und das Überwachungssystem erfasste innerhalb der ersten Woche zwei zusätzliche Legacy-Dienste, die noch veraltete Protokolle nutzten.
