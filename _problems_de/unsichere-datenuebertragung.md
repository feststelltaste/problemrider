---
title: Unsichere Datenübertragung
description: Sensible Daten werden ohne ordentliche Verschlüsselung oder Sicherheitskontrollen
  übertragen, was sie Abfangen und unbefugtem Zugriff aussetzt.
category:
- Security
- Security
related_problems:
- slug: secret-management-problems
  similarity: 0.55
- slug: error-message-information-disclosure
  similarity: 0.55
- slug: password-security-weaknesses
  similarity: 0.5
- slug: authentication-bypass-vulnerabilities
  similarity: 0.5
solutions:
- secret-management
- security-hardening-process
- authentication
- checksums
- prepared-statements
- privacy-by-design
- secure-protocols
- service-mesh
- certificate-management
- cryptographic-methods
- data-flow-control
- defense-lines
- digital-signatures
- encryption
- error-correction-codes
- key-management
- network-segmentation
- output-encoding
- zero-trust-architecture
layout: problem
lang: de
en_slug: insecure-data-transmission
---

## Description

Unsichere Datenübertragung tritt auf, wenn sensible Informationen über Netzwerke ohne angemessene Verschlüsselung oder Sicherheitskontrollen gesendet werden, was sie anfällig für Abfangen, Abhören und Man-in-the-Middle-Angriffe macht. Dies umfasst die Übertragung von Daten über unverschlüsselte Kanäle, die Nutzung schwacher Verschlüsselungsmethoden oder das Versäumnis, sichere Verbindungen ordentlich zu validieren.

## Indicators ⟡

- Sensible Daten werden über HTTP statt HTTPS übertragen
- Anwendungen akzeptieren ungültige oder selbstsignierte SSL-Zertifikate
- Schwache Verschlüsselungsalgorithmen oder -protokolle werden für die Datenübertragung genutzt
- Authentifizierungsdaten werden im Klartext gesendet
- Persönliche oder finanzielle Informationen werden ohne Verschlüsselung übertragen

## Symptoms ▲

- [Datenschutzrisiko](datenschutzrisiko.md)
<br/>  Die Übertragung von Daten ohne Verschlüsselung schafft direkt regulatorische und rechtliche Datenschutzrisiken.
- [Schwachstellen zur Umgehung der Authentifizierung](schwachstellen-zur-umgehung-der-authentifizierung.md)
<br/>  Unverschlüsselte Übertragung von Anmeldedaten ermöglicht Abfang- und Replay-Angriffe, die die Authentifizierung umgehen.
- [Stille Datenkorruption](stille-datenkorruption.md)
<br/>  Man-in-the-Middle-Angriffe auf unverschlüsselte Kanäle können Daten während der Übertragung ohne Erkennung modifizieren.
- [Regulatorische Compliance-Drift](regulatorische-compliance-drift.md)
<br/>  Unsichere Datenübertragung lässt das System aus der Compliance mit Sicherheitsvorschriften wie PCI-DSS und DSGVO fallen.
- [Kundenunzufriedenheit](kundenunzufriedenheit.md)
<br/>  Sicherheitsverletzungen durch unsichere Übertragung untergraben Kundenvertrauen und -zufriedenheit.

## Causes ▼

- [Veraltete Technologien](veraltete-technologien.md)
<br/>  Legacy-Systeme, die veraltete Protokolle nutzen, unterstützen möglicherweise keine modernen Verschlüsselungsstandards.
- [Unzureichende Design-Fähigkeiten](unzureichende-design-faehigkeiten.md)
<br/>  Entwicklern ohne Sicherheitsdesign-Wissen versäumen es möglicherweise, ordentliche Verschlüsselung für Daten während der Übertragung zu implementieren.
- [Konfigurationschaos](konfigurationschaos.md)
<br/>  Schlechtes Konfigurationsmanagement kann dazu führen, dass SSL/TLS in bestimmten Umgebungen falsch konfiguriert oder deaktiviert ist.

## Detection Methods ○

- **Netzwerkverkehrsanalyse:** Überwachung von Netzwerkkommunikation auf unverschlüsselte sensible Daten
- **SSL/TLS-Konfigurationstests:** Testen der Verschlüsselungsimplementierung und Zertifikatsvalidierung
- **Mixed-Content-Erkennung:** Identifikation von HTTPS-Seiten, die HTTP-Ressourcen laden
- **Protokollanalyse:** Analyse, welche Verschlüsselungsprotokolle und Cipher-Suiten genutzt werden
- **Zertifikatsvalidierungstests:** Testen des Anwendungsverhaltens mit ungültigen Zertifikaten

## Examples

Eine E-Commerce-Website sammelt Kreditkarteninformationen über HTTPS, übermittelt sie aber über HTTP an den Zahlungsabwickler. Während das anfängliche Formular für Nutzer sicher erscheint, werden die tatsächlichen Zahlungsdaten im Klartext übertragen, was sie anfällig für Abfangen macht. Netzwerkanalyse zeigt, dass Kreditkartennummern, Ablaufdaten und CVV-Codes für jeden sichtbar sind, der den Netzwerkverkehr überwacht. Ein weiteres Beispiel betrifft eine mobile Banking-Anwendung, die SSL-Zertifikate während der Entwicklung validiert, aber die Zertifikatsvalidierung in Produktion deaktiviert, um Konnektivitätsprobleme mit Load Balancern zu vermeiden. Dies macht die Anwendung anfällig für Man-in-the-Middle-Angriffe, bei denen Angreifer Banking-Transaktionen abfangen und modifizieren können, indem sie gefälschte Zertifikate präsentieren, die die Anwendung ohne Validierung akzeptiert.
