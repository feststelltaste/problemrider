---
title: Schwachstellen bei der Passwortsicherheit
description: Schwache Passwort-Richtlinien, unzureichende Speichermechanismen und
  schlechte Authentifizierungspraktiken schaffen Sicherheitslücken.
category:
- Security
related_problems:
- slug: authentication-bypass-vulnerabilities
  similarity: 0.6
- slug: secret-management-problems
  similarity: 0.6
- slug: session-management-issues
  similarity: 0.6
- slug: authorization-flaws
  similarity: 0.55
- slug: insecure-data-transmission
  similarity: 0.5
- slug: insufficient-audit-logging
  similarity: 0.5
solutions:
- security-hardening-process
- authentication
- authorization
- raising-user-awareness
- role-based-access-control
- secure-by-default
- secure-session-management
- security-policies-for-users
- cryptographic-methods
- encryption
- federated-identity
- key-management
- least-privilege
- secure-software
- two-factor-authentication
layout: problem
lang: de
en_slug: password-security-weaknesses
---

## Description

Schwachstellen bei der Passwortsicherheit treten auf, wenn Systeme unzureichende Passwort-Richtlinien implementieren, unsichere Speichermethoden nutzen oder schlechte Passwortverwaltungspraktiken haben. Diese Schwachstellen machen Nutzerkonten anfällig für Brute-Force-Angriffe, Wörterbuchangriffe, Credential Stuffing und unautorisierten Zugriff durch kompromittierte oder schwache Passwörter.

## Indicators ⟡

- Systeme erlauben schwache oder leicht zu erratende Passwörter
- Passwörter im Klartext oder mit schwachen Hashing-Algorithmen gespeichert
- Keine Konto-Sperrmechanismen für fehlgeschlagene Anmeldeversuche
- Passwort-Reset-Prozesse, die leicht ausnutzbar sind
- Standard- oder gemeinsam genutzte Passwörter über Systeme oder Konten hinweg

## Symptoms ▲

- [Schwachstellen zur Umgehung der Authentifizierung](schwachstellen-zur-umgehung-der-authentifizierung.md)
<br/>  Schwache Passwörter und schlechte Authentifizierungspraktiken machen es für Angreifer trivial, Authentifizierung durch Brute-Force oder Credential Stuffing zu umgehen.
- [Datenschutzrisiko](datenschutzrisiko.md)
<br/>  Schwache Passwortsicherheit setzt Nutzerkonten unautorisiertem Zugriff aus, was Risiken für den Schutz personenbezogener Daten und die regulatorische Compliance schafft.
- [Probleme im Session-Management](probleme-im-session-management.md)
<br/>  Schwache Passwortsicherheit kombiniert mit schlechter Session-Handhabung verschärft Schwachstellen, da kompromittierte Anmeldedaten dauerhaften unautorisierten Zugriff gewähren.
- [Kundenunzufriedenheit](kundenunzufriedenheit.md)
<br/>  Kontokompromittierungen infolge schwacher Passwortsicherheit untergraben das Nutzervertrauen und führen zu Kundenbeschwerden und Abwanderung.

## Causes ▼

- [Veraltete Technologien](veraltete-technologien.md)
<br/>  Legacy-Systeme nutzen möglicherweise veraltete Hashing-Algorithmen und Authentifizierungsmuster, die modernen Sicherheitsstandards vorausgehen.
- [Unerfahrene Entwickler](unerfahrene-entwickler.md)
<br/>  Entwickler ohne Sicherheitsexpertise implementieren möglicherweise naive Passwortspeicherung und -validierung, ohne die Schwachstellen zu verstehen.
- [Qualitätskompromisse](qualitaetskompromisse.md)
<br/>  Unter dem Druck, schnell zu liefern, werden ordentliche Passwortsicherheitspraktiken zugunsten einfacherer, aber unsicherer Implementierungen übersprungen.
- [Probleme beim Secret Management](probleme-beim-secret-management.md)
<br/>  Schlechte allgemeine Secret-Management-Praktiken erstrecken sich auf die Passwortbehandlung, mit unzureichendem Schutz gespeicherter Anmeldedaten.

## Detection Methods ○

- **Passwort-Richtlinien-Analyse:** Überprüfung von Passwortanforderungen und Durchsetzungsmechanismen
- **Passwortspeicher-Audit:** Untersuchung, wie Passwörter gehasht und in Datenbanken gespeichert werden
- **Brute-Force-Tests:** Testen der Systemresistenz gegen automatisierte Passwortangriffe
- **Sicherheitstests des Passwort-Resets:** Analyse des Passwort-Reset-Prozesses auf Schwachstellen
- **Scanning nach Standard-Anmeldedaten:** Überprüfung auf Systeme, die Standard- oder gängige Passwörter nutzen

## Examples

Eine Webanwendung speichert Nutzerpasswörter mittels MD5-Hashing ohne Salt. Wenn die Datenbank kompromittiert wird, nutzen Angreifer Rainbow Tables, um die MD5-Hashes schnell umzukehren und ursprüngliche Passwörter für die meisten Nutzer wiederherzustellen. Die Anwendung erlaubt auch Passwörter so einfach wie „123456" und implementiert keine Konto-Sperrung nach fehlgeschlagenen Anmeldeversuchen, was Brute-Force-Angriffe trivial macht. Ein weiteres Beispiel betrifft ein Unternehmenssystem, das mit Standard-Administratoranmeldedaten „admin/admin" ausgeliefert wird, und viele Installationen ändern diese Standardwerte nie. Angreifer nutzen automatisierte Scanner, um Systeme mit Standard-Anmeldedaten zu finden und administrativen Zugriff zu erlangen. Die Passwort-Reset-Funktionalität sendet neue Passwörter im Klartext per E-Mail, was eine weitere Schwachstelle schafft, bei der das Abfangen von E-Mails Konten kompromittieren kann.
