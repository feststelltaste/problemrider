---
title: Verschlüsselung
description: Verschlüsselung von Daten bei Übertragung und Speicherung.
category:
- Security
problems:
- insecure-data-transmission
- data-protection-risk
- password-security-weaknesses
- regulatory-compliance-drift
- secret-management-problems
- error-message-information-disclosure
- session-management-issues
layout: solution
lang: de
en_slug: encryption
related_solutions:
- slug: cryptographic-methods
  similarity: 0.85
- slug: authentication
  similarity: 0.8
- slug: key-management
  similarity: 0.8
- slug: secret-management
  similarity: 0.8
- slug: backup-and-recovery
  similarity: 0.8
- slug: secure-protocols
  similarity: 0.75
---

## Description

Verschlüsselung transformiert Daten in eine ohne den korrekten Schlüssel unlesbare Form und schützt sie sowohl während der Übertragung über Netzwerke als auch im Ruhezustand im Speicher, sodass ihre Vertraulichkeit nicht mehr vollständig davon abhängt, dass Netzwerkperimeter-Sicherheit oder physische Zugriffskontrollen perfekt halten. Legacy-Systeme wurden häufig in Ären gebaut, in denen perimeterbasiertes Vertrauen als ausreichend galt, was unverschlüsselte HTTP-Verbindungen, Klartext-Datenbankverbindungen, unverschlüsselte Dateiübertragungen und Klartext-Konfigurationsdateien zu routinemäßigen, unbemerkten Designentscheidungen macht, statt zu bewussten Risikoakzeptanzen. Verschlüsselung nachträglich in ein solches System einzubauen bedeutet typischerweise, Maßnahmen einzuführen, die das ursprüngliche Design nie antizipierte: TLS-Terminierungs-Proxies vor Komponenten, die modernes TLS nicht nativ unterstützen können, transparente oder spaltenebenenbasierte Verschlüsselung, hinzugefügt zu Datenbanken, die nie damit gebaut wurden, und dedizierte Schlüsselverwaltungssysteme, um Schlüssel getrennt von den Daten zu halten, die sie schützen. Weil Legacy-Systeme oft sensible Daten tragen — Gesundheitsakten, Zahlungsdetails, persönliche Informationen —, angehäuft über lange Betriebshistorien, ist Verschlüsselung häufig das, was die Lücke zwischen tatsächlicher Praxis und regulatorischen Anforderungen wie PCI DSS, HIPAA oder DSGVO schließt, und sie bietet eine Schicht von Defense in Depth, die Daten weiterhin schützt, selbst wenn ein Netzwerk kompromittiert oder Speichermedien gestohlen werden. Ihre Kosten sind konkret statt hypothetisch: Rechenoverhead auf Hardware, die ursprünglich nicht für kryptografische Arbeit dimensioniert war, Schlüsselverwaltung, die zu einem kritischen Single Point of Failure wird, und verschlüsselte Daten, die ohne zusätzlichen Engineering-Aufwand schwerer zu durchsuchen, zu indizieren und zu debuggen sind.

## How to Apply ◆

> Legacy-Systeme übertragen und speichern häufig sensible Daten ohne Verschlüsselung und verlassen sich auf Netzwerkperimeter-Sicherheit oder physische Zugriffskontrollen, die für moderne Bedrohungsmodelle unzureichend sind. Die Implementierung von Verschlüsselung schützt Daten sowohl während der Übertragung als auch im Ruhezustand.

- Prüfen Sie alle Datenübertragungspfade im Legacy-System und identifizieren Sie unverschlüsselte Kanäle: HTTP-Verbindungen, unverschlüsselte Datenbankverbindungen, Klartext-Dateiübertragungen (FTP), unverschlüsselte serviceübergreifende Kommunikation und Klartext-E-Mails mit sensiblen Anhängen.
- Aktivieren Sie TLS 1.2 oder höher für jede Netzwerkkommunikation. Deaktivieren Sie ältere Protokolle (SSL 3.0, TLS 1.0, TLS 1.1) und konfigurieren Sie starke Cipher Suites. Für Legacy-Systeme, die modernes TLS nicht unterstützen können, implementieren Sie einen TLS-terminierenden Reverse Proxy vor der Legacy-Komponente.
- Implementieren Sie Verschlüsselung im Ruhezustand für Datenbanken mit sensiblen Daten. Nutzen Sie Transparent Data Encryption (TDE) für Verschlüsselung auf Datenbankebene oder spaltenebenenbasierte Verschlüsselung für spezifische sensible Felder (Kreditkartennummern, Sozialversicherungsnummern, Gesundheitsakten).
- Verschlüsseln Sie Backup-Dateien und Archive, besonders wenn sie auf Wechselmedien gespeichert oder an Off-Site-Speicher übertragen werden. Unverschlüsselte Backups sind eine häufige Quelle für Datenschutzverletzungen.
- Implementieren Sie Schlüsselverwaltung, die Verschlüsselungsschlüssel von verschlüsselten Daten trennt. Speichern Sie Schlüssel in dedizierten Schlüsselverwaltungssystemen (KMS), Hardware-Sicherheitsmodulen (HSMs) oder Secret-Management-Diensten, niemals neben den Daten, die sie schützen.
- Verschlüsseln Sie Konfigurationsdateien und Umgebungsvariablen, die sensible Werte enthalten (Datenbankpasswörter, API-Schlüssel, Service-Account-Anmeldedaten). Legacy-Systeme speichern diese üblicherweise im Klartext.
- Implementieren Sie feldebenenbasierte Verschlüsselung für die sensibelsten Datenelemente und stellen Sie sicher, dass selbst Datenbankadministratoren und Systemoperatoren ohne expliziten Schlüsselzugriff nicht auf Klartextwerte zugreifen können.

## Tradeoffs ⇄

> Verschlüsselung schützt die Vertraulichkeit von Daten, selbst wenn andere Sicherheitskontrollen versagen, fügt aber Rechenoverhead und Schlüsselverwaltungskomplexität hinzu und kann Debugging und Überwachung erschweren.

**Vorteile:**

- Schützt die Vertraulichkeit von Daten, selbst wenn das Netzwerk kompromittiert, Speichermedien gestohlen oder Datenbankzugriffskontrollen umgangen werden.
- Erfüllt regulatorische Anforderungen (PCI DSS, HIPAA, DSGVO), die Verschlüsselung sensibler Daten bei Übertragung und Speicherung vorschreiben.
- Bietet Defense in Depth — selbst wenn ein Angreifer Zugriff auf das System erhält, bleiben verschlüsselte Daten ohne die Entschlüsselungsschlüssel geschützt.
- Ermöglicht sichere Nutzung von Cloud-Speicher und Drittanbieter-Infrastruktur für Legacy-System-Daten, indem sichergestellt wird, dass Daten unabhängig von der Sicherheit des Speicheranbieters geschützt bleiben.

**Kosten und Risiken:**

- Verschlüsselung fügt jeder Datenoperation CPU-Overhead hinzu, was die Performance auf Legacy-Hardware beeinträchtigen kann, die nicht für kryptografische Verarbeitung dimensioniert war.
- Fehler in der Schlüsselverwaltung (verlorene Schlüssel, kompromittierte Schlüssel, nicht verfügbares KMS) können zu dauerhaftem Datenverlust oder weitreichender Exposition führen.
- Verschlüsselte Daten können ohne Entschlüsselung nicht durchsucht, indiziert oder verarbeitet werden, was möglicherweise Anwendungsänderungen erfordert und die Abfrageperformance beeinträchtigt.
- Debugging und Überwachung werden schwieriger, wenn Daten verschlüsselt sind, da Log-Analyse und Dateninspektion zusätzliche Schritte erfordern.

## How It Could Be

> Die folgenden Szenarien veranschaulichen, wie Verschlüsselung Legacy-System-Daten schützt.

Ein Legacy-HR-System überträgt Gehaltsdaten von Mitarbeitern zwischen dem Hauptanwendungsserver und einem Berichtsserver über eine unverschlüsselte HTTP-Verbindung im internen Netzwerk. Ein Netzwerksicherheitsaudit zeigt, dass jeder mit Zugriff auf das interne Netzwerk diesen Traffic mittels Standard-Packet-Sniffing-Werkzeugen erfassen kann. Das Team implementiert TLS für die Verbindung zwischen den beiden Servern, deployt einen Reverse Proxy mit TLS-Terminierung vor der Legacy-Anwendung (die HTTPS nicht nativ unterstützt) und fügt der Berichtsdatenbank TDE hinzu. Zusätzlich verschlüsseln sie die nächtlichen Datenexportdateien, die an den Lohnabrechnungsanbieter übertragen werden, und ersetzen eine unverschlüsselte FTP-Übertragung durch SFTP. Das gesamte Netzwerk trägt jetzt nur noch verschlüsselten Traffic, und ein nachfolgender Penetrationstest bestätigt, dass erfasste Netzwerkpakete keine lesbaren sensiblen Daten offenbaren.

Eine Legacy-Kundendatenbank speichert Kreditkartennummern im Klartext, um ein vor 10 Jahren gebautes Feature für wiederkehrende Abrechnung zu unterstützen. Ein PCI-DSS-Audit identifiziert dies als kritischen Befund, der sofortige Abhilfe erfordert. Das Team implementiert spaltenebenenbasierte Verschlüsselung mittels AES-256 für das Kreditkartennummernfeld, speichert Verschlüsselungsschlüssel in einem dedizierten KMS mit auf das Abrechnungs-Service-Konto beschränktem Zugriff und modifiziert die Anwendung, um Werte nur am Nutzungspunkt während Abrechnungsoperationen zu entschlüsseln. Sie implementieren auch Tokenisierung für Anzeigezwecke — die Anwendung zeigt in der Nutzeroberfläche nur die letzten vier Ziffern der Kartennummer. Die tokenisierte Anzeige und verschlüsselte Speicherung reduzieren den PCI-DSS-Compliance-Umfang von der gesamten Anwendung auf nur den Abrechnungsservice, was künftige Audits erheblich vereinfacht.
