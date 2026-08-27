---
title: Zwei-Faktor-Authentifizierung
description: Überprüfung der Identität mithilfe zweier unabhängiger
  Faktoren.
category:
- Security
problems:
- authentication-bypass-vulnerabilities
- password-security-weaknesses
- data-protection-risk
- session-management-issues
- authorization-flaws
- regulatory-compliance-drift
layout: solution
lang: de
en_slug: two-factor-authentication
related_solutions:
- slug: authentication
  similarity: 0.8
- slug: security-policies-for-users
  similarity: 0.75
- slug: cryptographic-methods
  similarity: 0.75
- slug: federated-identity
  similarity: 0.75
- slug: raising-user-awareness
  similarity: 0.7
- slug: secure-protocols
  similarity: 0.7
---

## Description

Zwei-Faktor-Authentifizierung verlangt von einem Nutzer, seine Identität mit zwei unabhängigen Arten von Nachweisen zu beweisen — typischerweise etwas, das er weiß, wie ein Passwort, plus etwas, das er hat oder ist, wie einen zeitbasierten Einmalcode, ein Hardware-Token oder eine Push-Benachrichtigung —, sodass ein kompromittiertes Passwort allein nicht mehr ausreicht, um Zugang zu erlangen. Dies zählt akut für Legacy-Systeme, weil ihre Authentifizierungsmechanismen häufig in einer Ära gebaut wurden, in der reines Passwort-Login die Norm war und Account-Übernahme-Techniken wie Credential Stuffing und Passwort-Wiederverwendung über kompromittierte Seiten hinweg weit weniger verbreitete Bedrohungen waren als heute, was diese Systeme mit einer Kontrolle zur Verteidigung hochwertigen Zugangs zurücklässt, die relativ zur aktuellen Angriffslandschaft schwach geworden ist. Da die Nachrüstung des Legacy-Authentifizierungscodes selbst invasiv und riskant sein kann, wird der zweite Faktor oft durch einen Authentifizierungs-Proxy oder einen externen Identitätsanbieter eingeschichtet, der vor dem Legacy-Login-Flow sitzt, was dem System erlaubt, modernen Schutz zu erlangen, ohne Änderungen an brüchiger, schlecht verstandener interner Authentifizierungslogik zu erfordern. Die Priorisierung des Rollouts zuerst auf die privilegiertesten Konten — Administratoren, Datenbankzugriff, Deployment-Zugangsdaten — zielt den zweiten Faktor auf die Zugangspunkte, an denen ein einzelnes gestohlenes Passwort sonst den größten Schaden verursachen würde. Die Praxis tauscht etwas Login-Reibung und Support-Overhead für verlorene zweite Faktoren gegen eine erhebliche Reduzierung des Risikos einer Kontokompromittierung ein, was üblicherweise ein günstiger Tausch für Systeme ist, die sonst nicht mit sich entwickelnden zugangsdatenbasierten Angriffen Schritt halten können.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken, um diese Lösung im Kontext eines Legacy-Systems umzusetzen.

- Bewerten Sie 2FA-Methoden, die für die Nutzerbasis des Legacy-Systems geeignet sind: TOTP-Apps, Hardware-Token, SMS-Codes oder Push-Benachrichtigungen
- Implementieren Sie 2FA zuerst für die privilegiertesten Konten (Administratoren, Datenbankzugriff, Deployment-Zugangsdaten)
- Fügen Sie 2FA-Unterstützung zum Legacy-Authentifizierungs-Flow hinzu, ohne die bestehende Login-Erfahrung zu stören
- Bieten Sie Fallback-Wiederherstellungsmechanismen wie Backup-Codes für Nutzer, die den Zugang zu ihrem zweiten Faktor verlieren
- Integrieren Sie mit bestehenden Identitätsanbietern oder implementieren Sie einen eigenständigen 2FA-Dienst, den die Legacy-Anwendung aufruft
- Bieten Sie eine Übergangsperiode, in der 2FA empfohlen wird, bevor es verpflichtend wird
- Protokollieren Sie alle 2FA-Ereignisse zu Audit-Zwecken und überwachen Sie auf anomale Authentifizierungsmuster

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie dafür aufgeben.

**Vorteile:**
- Reduziert das Risiko der Kontokompromittierung durch gestohlene oder schwache Passwörter dramatisch
- Bietet eine starke zusätzliche Verteidigungsschicht für kritischen Systemzugang
- Erfüllt regulatorische und Compliance-Anforderungen für starke Authentifizierung
- Schreckt automatisiertes Credential Stuffing und Brute-Force-Angriffe ab

**Kosten und Risiken:**
- Fügt dem Login-Prozess Reibung hinzu, was Nutzer frustrieren und die Produktivität reduzieren kann
- Verlorene oder fehlfunktionierende zweite Faktoren können Nutzer aussperren, was Support-Prozesse erfordert
- SMS-basierte 2FA ist anfällig für SIM-Swapping und Abfangangriffe
- Die Nachrüstung von 2FA in Legacy-Authentifizierungssysteme könnte erhebliche Codeänderungen erfordern
- Service-Konten und automatisierte Prozesse könnten 2FA-Workflows nicht leicht unterstützen

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Eine Anwaltskanzlei erlebte eine Sicherheitsverletzung, als ein Angreifer Zugangsdaten nutzte, die aus einem Datenleck auf einer anderen Seite stammten, um auf ihr Legacy-Fallverwaltungssystem zuzugreifen. Nach dem Vorfall implementierte die Kanzlei TOTP-basierte Zwei-Faktor-Authentifizierung für alle Nutzer. Für die Legacy-Anwendung, die 2FA nicht nativ unterstützte, setzten sie einen Authentifizierungs-Proxy ein, der den zweiten Faktor handhabte, bevor er authentifizierte Sitzungen an das Legacy-System weiterleitete. Dieser Ansatz erforderte keine Änderungen an der Legacy-Codebasis. Innerhalb von drei Monaten wurden zwei zusätzliche Credential-Stuffing-Versuche durch die 2FA-Anforderung blockiert, was ihre Wirksamkeit bestätigte.
