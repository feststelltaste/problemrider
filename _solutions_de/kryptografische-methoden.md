---
title: Kryptografische Methoden
description: Nutzung erprobter und standardisierter Algorithmen und Protokolle für
  kryptografische Funktionen.
category:
- Security
problems:
- insecure-data-transmission
- password-security-weaknesses
- data-protection-risk
- regulatory-compliance-drift
- secret-management-problems
- authentication-bypass-vulnerabilities
layout: solution
lang: de
en_slug: cryptographic-methods
related_solutions:
- slug: encryption
  similarity: 0.85
- slug: secure-protocols
  similarity: 0.85
- slug: authentication
  similarity: 0.8
- slug: secret-management
  similarity: 0.8
- slug: key-management
  similarity: 0.8
- slug: patch-management
  similarity: 0.75
---

## Description

Kryptografische Methoden als Lösung bedeutet, die tatsächlich von einem Legacy-System genutzten Verschlüsselungs-, Hashing- und Zufallszahlengenerierungsmechanismen zu prüfen und durch erprobte, geprüfte, standardisierte Algorithmen und etablierte Bibliotheken zu ersetzen, statt weiterhin auf das zu vertrauen, was zum Zeitpunkt des ursprünglichen Systembaus als angemessen galt. Legacy-Systeme betreiben häufig noch veraltete Algorithmen — DES, MD5, SHA-1, RC4 — oder maßgeschneiderte Verschlüsselungsschemata, die nie ernsthafter kryptografischer Prüfung unterzogen wurden, einfach weil niemand die Kryptografie seit der ursprünglichen Implementierung überarbeitet hat, und diese Entscheidungen bieten nur den Anschein von Sicherheit gegenüber moderner Rechenleistung und Angriffstechniken. Das Risiko verschärft sich, wenn Legacy-Code selbst einen starken Algorithmus falsch implementiert — die Nutzung des ECB-Modus, der Datenmuster offenbart, oder das Seeden von Tokens mit einem nicht-kryptografischen Zufallszahlengenerator wie Javas `Math.random()` —, da solche Implementierungsdetailfehler den Sicherheitsnutzen eines ansonsten soliden Algorithmus vollständig zunichtemachen können. Die Migration zu aktuellen Standards (AES-256, SHA-256/SHA-3, bcrypt/Argon2id, CSPRNGs) mittels etablierter Bibliotheken wie OpenSSL oder libsodium statt eigenem Code adressiert beide Probleme gleichzeitig: Sie ersetzt schwache Algorithmen und entfernt das Risiko einer subtil fehlerhaften Eigenimplementierung von Grund auf. Weil die Migration bereits verschlüsselter Daten ein Zeitfenster erfordert, in dem sie unter dem alten Schema entschlüsselt existieren, bevor sie unter dem neuen wieder verschlüsselt werden, und weil Algorithmusänderungen Integrationen brechen können, die das Legacy-Format erwarten, wird diese Modernisierung typischerweise als sorgfältig gestufte rollierende Migration statt als sofortiger Umstieg durchgeführt.

## How to Apply ◆

> Legacy-Systeme nutzen oft veraltete kryptografische Algorithmen (DES, MD5, SHA-1, RC4) oder maßgeschneiderte Verschlüsselungsschemata, die falsche Sicherheit bieten. Die Modernisierung kryptografischer Methoden stellt sicher, dass Datenschutz auf erprobten, geprüften Algorithmen beruht.

- Prüfen Sie die gesamte kryptografische Nutzung im Legacy-System: Passwort-Hashing, Datenverschlüsselung im Ruhezustand und bei der Übertragung, digitale Signaturen, Token-Generierung und Zufallszahlengenerierung. Identifizieren Sie, welche Algorithmen und Schlüsselgrößen genutzt werden.
- Ersetzen Sie veraltete Algorithmen durch aktuelle Standards: AES-256 für symmetrische Verschlüsselung, RSA-2048+ oder ECDSA P-256+ für asymmetrische Operationen, SHA-256 oder SHA-3 für Hashing und bcrypt/Argon2id für Passwort-Hashing.
- Nutzen Sie etablierte kryptografische Bibliotheken (OpenSSL, libsodium, Bouncy Castle) statt eigener Implementierungen. Selbst bekannte Algorithmen können falsch implementiert werden, was zu schwer erkennbaren Schwachstellen führt.
- Implementieren Sie ordnungsgemäße Zufallszahlengenerierung mittels kryptografisch sicherer Pseudozufallszahlengeneratoren (CSPRNGs) für alle sicherheitsrelevanten Operationen: Session-Tokens, API-Schlüssel, Nonces, Initialisierungsvektoren und Passwort-Salts.
- Migrieren Sie vom ECB-Modus (Electronic Codebook) zu authentifizierten Verschlüsselungsmodi wie AES-GCM oder ChaCha20-Poly1305, die sowohl Vertraulichkeit als auch Integritätsschutz bieten. Der ECB-Modus, häufig in Legacy-Systemen gefunden, offenbart Muster in verschlüsselten Daten.
- Planen und implementieren Sie Krypto-Agilität: Gestalten Sie das System so, dass kryptografische Algorithmen ohne größere Codeänderungen ersetzt werden können. Dies bereitet auf zukünftige Algorithmus-Veralterung und eventuelle Post-Quanten-Kryptografie-Migration vor.
- Stellen Sie sicher, dass alle kryptografischen Operationen ordnungsgemäße Schlüsselableitung, Padding und Initialisierungsvektor-Handhabung nutzen. Viele Legacy-Schwachstellen entstehen aus diesen Implementierungsdetails statt aus dem Kernalgorithmus selbst.

## Tradeoffs ⇄

> Standardisierte kryptografische Methoden bieten gut geprüften Datenschutz, der bekannten Angriffen standhält, aber die Migration von Legacy-Algorithmen erfordert sorgfältige Planung, um Datenverlust oder Zugriffsunterbrechung zu vermeiden.

**Vorteile:**

- Schützt Daten gegen bekannte Angriffe, die Schwächen in veralteten Algorithmen ausnutzen, und erhält Vertraulichkeit und Integrität.
- Stellt Compliance mit aktuellen Sicherheitsstandards und Vorschriften sicher, die spezifische kryptografische Anforderungen vorschreiben.
- Nutzt Jahrzehnte kryptografischer Forschung und Peer Review, statt sich auf Sicherheit durch Obskurität zu verlassen.
- Ermöglicht Interoperabilität mit modernen Systemen und Protokollen, die aktuelle kryptografische Standards erfordern.

**Kosten und Risiken:**

- Die Migration verschlüsselter Daten von Legacy-Algorithmen zu modernen erfordert Entschlüsselung mit dem alten Algorithmus und Neuverschlüsselung mit dem neuen, was ein Zeitfenster der Exposition schafft.
- Stärkere Algorithmen können höheren Rechenaufwand haben, was die Performance auf Legacy-Hardware beeinträchtigt.
- Kryptografische Migration kann Integrationen mit externen Systemen brechen, die den Legacy-Algorithmus oder das Legacy-Datenformat erwarten.
- Falsche Implementierung selbst starker Algorithmen (falscher Modus, vorhersehbare IVs, unsachgemäßes Padding) kann ihren Sicherheitsnutzen zunichtemachen.

## How It Could Be

> Die folgenden Szenarien veranschaulichen, wie die Aktualisierung kryptografischer Methoden die Sicherheit von Legacy-Systemen stärkt.

Ein Legacy-Gesundheitssystem speichert Sozialversicherungsnummern von Patienten, verschlüsselt mit DES (56-Bit-Schlüssel), was Standard war, als das System 1998 gebaut wurde. Moderne Hardware kann DES in Stunden brute-forcen. Das Team implementiert eine rollierende Migration: Sie fügen eine neue AES-256-GCM-verschlüsselte Spalte hinzu, schreiben einen Batch-Prozess, der jeden Wert mit DES entschlüsselt und mit AES-256-GCM neu verschlüsselt, und aktualisieren die Anwendung, aus der neuen Spalte zu lesen. Die Migration läuft während Wartungsfenstern über zwei Wochen und verarbeitet 2 Millionen Datensätze. Nach der Verifikation wird die DES-verschlüsselte Spalte sicher gelöscht. Die Anwendung wird auch aktualisiert, um einen Schlüsselverwaltungsdienst statt eines fest codierten, im Quellcode eingebetteten Verschlüsselungsschlüssels zu nutzen.

Eine Legacy-Bankanwendung generiert Session-Tokens mit Javas `Math.random()`, was nicht kryptografisch sicher ist und vorhersehbare Sequenzen erzeugt. Ein Sicherheitsforscher demonstriert, dass er durch Beobachtung einiger hundert Tokens zukünftige Tokens mit hoher Genauigkeit vorhersagen kann, was Session-Hijacking ermöglicht. Das Team ersetzt `Math.random()` durch `SecureRandom` mit dem nativen CSPRNG der Plattform und erhöht die Token-Länge von 32 Bit auf 256 Bit. Sie fügen auch Token-Bindung an die TLS-Session des Clients hinzu, um Token-Replay von unterschiedlichen Verbindungen zu verhindern. Nach der Korrektur bestätigt Penetrationstesting, dass Token-Vorhersage rechnerisch nicht machbar ist.
