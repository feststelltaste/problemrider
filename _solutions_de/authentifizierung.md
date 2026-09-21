---
title: Authentifizierung
description: Überprüfung der Identität von Nutzern und Systemen.
category:
- Security
problems:
- authentication-bypass-vulnerabilities
- password-security-weaknesses
- session-management-issues
- data-protection-risk
- authorization-flaws
- error-message-information-disclosure
- insecure-data-transmission
layout: solution
lang: de
en_slug: authentication
related_solutions:
- slug: cryptographic-methods
  similarity: 0.8
- slug: authorization
  similarity: 0.8
- slug: encryption
  similarity: 0.8
- slug: logging-and-monitoring
  similarity: 0.8
- slug: two-factor-authentication
  similarity: 0.8
- slug: secret-management
  similarity: 0.75
---

## Description

Authentifizierung ist der Mechanismus, mit dem ein System überprüft, dass ein Nutzer oder ein anderes System tatsächlich das ist, was es vorgibt zu sein, typischerweise indem eine vorgelegte Anmeldeinformation (Passwort, Token, Zertifikat oder biometrischer Faktor) gegen eine sicher gespeicherte Referenz geprüft wird und eine Sitzung oder ein Token ausgestellt wird, das für diese verifizierte Identität in nachfolgenden Anfragen steht. Ihre Korrektheit hängt vollständig von Implementierungsdetails ab — der Stärke des Passwort-Hashing-Algorithmus, der Unvorhersagbarkeit von Sitzungs-Tokens, dem Vorhandensein von Rate Limiting und Kontosperrung — Details, die genau dort liegen, wo Legacy-Systeme dazu tendieren, am weitesten von aktueller Praxis abgedriftet zu sein. Vor Jahren oder Jahrzehnten geschriebene, selbstgebaute Authentifizierungslogik stammt häufig aus der Zeit vor modernen Hashing-Standards, nutzt sequenzielle oder anderweitig erratbare Sitzungskennungen oder leckt Informationen durch inkonsistente Fehlermeldungen — nichts davon wurde damals als ernstes Risiko betrachtet, aber alles davon sind heute gut verstandene Angriffsvektoren. Weil Authentifizierung die Torwächterschicht ist, hinter der jede andere Sicherheitskontrolle sitzt, untergraben Schwächen hier Autorisierung, Audit-Logging und Datenschutz, unabhängig davon, wie gut diese anderen Schichten gebaut sind. Die Modernisierung der Authentifizierung im Legacy-Kontext ist ebenso sehr ein Migrationsproblem wie ein Sicherheitsproblem: Bestehende Anmeldeinformationen, Sitzungen und Integrationen müssen weiterhin funktionieren, während der zugrunde liegende Mechanismus darunter ersetzt wird, üblicherweise durch Techniken wie transparentes Re-Hashing beim Login statt eines erzwungenen, disruptiven Umstiegs.

## How to Apply ◆

> Legacy-Systeme verlassen sich häufig auf veraltete Authentifizierungsmechanismen — Klartext-Passwörter, schwache Hashing-Algorithmen oder selbstgebaute Authentifizierungslogik mit bekannten Schwachstellen. Die Modernisierung der Authentifizierung ist ein grundlegender Schritt zur Absicherung jedes Legacy-Systems.

- Auditieren Sie den bestehenden Authentifizierungsmechanismus, um Schwächen zu identifizieren: Klartext- oder schwach gehashte Passwörter (MD5, SHA-1 ohne Salt), fest codierte Anmeldeinformationen, Sitzungs-Tokens mit vorhersagbaren Mustern und Authentifizierungs-Umgehungspfade.
- Ersetzen Sie selbstgebaute Authentifizierungsimplementierungen durch gut getestete Authentifizierungsbibliotheken oder Frameworks. Legacy-Systeme enthalten oft handgeschriebenen Authentifizierungscode mit subtilen Schwachstellen, die Standardbibliotheken bereits behoben haben.
- Implementieren Sie starkes Passwort-Hashing mit bcrypt, scrypt oder Argon2id mit angemessenen Arbeitsfaktoren. Migrieren Sie bestehende Passwort-Hashes durch Re-Hashing beim nächsten erfolgreichen Login — Nutzer authentifizieren sich mit dem alten Hash, und ihr Passwort wird sofort mit dem modernen Algorithmus neu gehasht.
- Fügen Sie Multi-Faktor-Authentifizierung (MFA) für administrative Konten und sensible Operationen hinzu. Selbst wenn die gesamte Nutzerbasis MFA nicht sofort übernehmen kann, eliminiert der Schutz privilegierter Konten die höchsten Authentifizierungsrisiken.
- Implementieren Sie Kontosperrung oder progressive Verzögerungen nach fehlgeschlagenen Login-Versuchen, um Brute-Force-Angriffe zu verhindern. Stellen Sie sicher, dass Sperrrichtlinien keine Dienstverweigerung ermöglichen, indem legitime Nutzer ausgesperrt werden — nutzen Sie CAPTCHA oder temporäre Verzögerungen statt permanenter Sperrung.
- Sichern Sie das Sitzungsmanagement, indem Sie kryptografisch zufällige Sitzungs-Tokens generieren, angemessene Ablaufzeiten setzen und Sitzungen bei Abmeldung und Passwortänderung invalidieren. Nutzen Sie sichere, HttpOnly-, SameSite-Cookie-Attribute.
- Eliminieren Sie generische Fehlermeldungen in Authentifizierungsabläufen („ungültiger Nutzername oder Passwort" statt „Nutzername nicht gefunden" oder „falsches Passwort"), um Nutzeraufzählung zu verhindern.

## Tradeoffs ⇄

> Starke Authentifizierung verhindert unautorisierten Zugriff und ist eine Voraussetzung für alle anderen Sicherheitskontrollen, fügt aber Reibung zur Nutzererfahrung und Komplexität zur Systemintegration hinzu.

**Vorteile:**

- Verhindert unautorisierten Zugriff, indem überprüft wird, dass Nutzer und Systeme das sind, was sie vorgeben zu sein, und bildet die Grundlage aller Zugriffskontrolle.
- Schützt gegen anmeldeinformationsbasierte Angriffe (Brute Force, Credential Stuffing, Phishing) durch modernes Hashing, MFA und Sperrrichtlinien.
- Bietet Verantwortlichkeit, indem jede Aktion im System mit einer authentifizierten Identität verknüpft wird.
- Ermöglicht Compliance mit Sicherheitsstandards und Vorschriften, die starke Authentifizierungskontrollen vorschreiben.

**Kosten und Risiken:**

- Stärkere Authentifizierung fügt Nutzerreibung hinzu (MFA-Schritte, Passwortkomplexitätsanforderungen, Sitzungs-Timeouts), was Akzeptanz und Produktivität verringern kann.
- Die Migration von Legacy-Authentifizierungsmechanismen erfordert sorgfältige Planung, um zu vermeiden, dass Nutzer während des Übergangs ausgesperrt werden.
- Die Integration mit externen Systemen, die auf den Legacy-Authentifizierungsmechanismus angewiesen sind, könnte brechen, wenn Authentifizierung modernisiert wird.
- Kontosperrmechanismen können für Dienstverweigerung ausgenutzt werden, wenn sie nicht mit Rate Limiting statt harter Sperrungen implementiert werden.

## How It Could Be

> Die folgenden Szenarien veranschaulichen, wie Authentifizierungsmodernisierung Sicherheitslücken in Legacy-Systemen angeht.

Eine Legacy-Unternehmensanwendung speichert Nutzerpasswörter als ungesaltene MD5-Hashes in der Datenbank. Ein Datenbank-Backup wird versehentlich exponiert, und ein Angreifer nutzt Rainbow Tables, um 85 % der Passwörter innerhalb von Stunden zu knacken. Das Team implementiert eine Migrationsstrategie: Sie fügen eine neue bcrypt-Passwortspalte zur Nutzertabelle hinzu. Wenn sich ein Nutzer erfolgreich mit seinem MD5-gehashten Passwort anmeldet, hasht das System sein Klartext-Passwort transparent mit bcrypt neu und speichert den neuen Hash. Nutzer, die sich innerhalb von 90 Tagen nicht angemeldet haben, werden gezwungen, ihre Passwörter zurückzusetzen. Nach sechs Monaten wurden 95 % der aktiven Konten zu bcrypt migriert, und die Legacy-MD5-Spalte wird entfernt. Das Team fügt außerdem MFA via TOTP für alle administrativen Konten hinzu, was verhindert, dass zukünftige Anmeldeinformations-Kompromittierungen administrativen Zugriff gewähren.

Ein Legacy-Lieferkettenmanagementsystem nutzt Sitzungs-IDs, die sequenzielle Ganzzahlen sind, was Session-Hijacking für jeden trivial macht, der eine gültige Sitzungs-ID beobachten oder erraten kann. Das Team ersetzt das Sitzungsmanagement durch kryptografisch zufällige 256-Bit-Sitzungs-Tokens, setzt den Sitzungsablauf auf 30 Minuten Inaktivität und implementiert Sitzungsbindung an die IP-Adresse und den User-Agent des Clients. Sie fügen außerdem einen Sitzungsinvalidierungs-Endpunkt hinzu, den die Anwendung bei Abmeldung aufruft. Nach dem Deployment bestätigt der Penetrationstest des Sicherheitsteams, dass Sitzungsvorhersage und Hijacking nicht mehr machbar sind, und die authentifizierte Sitzung wird bei Abmeldung ordentlich beendet.
