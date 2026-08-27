---
title: Rollenbasierte Zugriffskontrolle
description: Steuerung des Zugriffs auf Anwendungskomponenten basierend auf
  Rollen.
category:
- Security
problems:
- authorization-flaws
- authentication-bypass-vulnerabilities
- data-protection-risk
- password-security-weaknesses
- session-management-issues
- regulatory-compliance-drift
- secret-management-problems
- authorization-role-explosion
layout: solution
lang: de
en_slug: role-based-access-control
related_solutions:
- slug: authorization
  similarity: 0.8
- slug: authorization-concept
  similarity: 0.75
- slug: domain-based-authorization-concept
  similarity: 0.75
- slug: security-policies-for-users
  similarity: 0.75
- slug: least-privilege
  similarity: 0.75
- slug: secure-by-default
  similarity: 0.7
---

## Description

Rollenbasierte Zugriffskontrolle (RBAC) ist ein Autorisierungsmodell, bei dem Berechtigungen Rollen gewährt werden, die Geschäftsfunktionen entsprechen — wie Schadensregulierer oder Systemadministrator —, statt direkt einzelnen Nutzern, sodass die Zugriffsrechte eines Nutzers aus den ihm zugewiesenen Rollen folgen, statt einzeln pro Berechtigung konfiguriert zu werden. Autorisierungsentscheidungen auf diese Weise zu zentralisieren ersetzt verstreute, inline eingebettete Berechtigungsprüfungen über die gesamte Anwendung durch eine einzige, konsistente Menge von Rollendefinitionen, die alle Komponenten konsultieren, was auch jede Zugriffsentscheidung an einem Ort auditierbar macht. Legacy-Systeme haben sich sehr häufig zum gegenteiligen Modell entwickelt: einzeln zugewiesene Berechtigungen, angesammelt Nutzer für Nutzer über viele Jahre, während jeder Neuzugang denselben Zugang erhielt wie die letzte Person oder Einzelfall-Ausnahmen geschaffen wurden, um eine spezifische Anfrage zu entblocken, was eine Berechtigungsstruktur hinterlässt, die niemand vollständig versteht und die unverhältnismäßigen administrativen Aufwand allein zur Pflege erfordert. Die Migration eines solchen Systems zu RBAC erfordert zunächst eine Inventarisierung dessen, was heute tatsächlich an Zugang existiert — was häufig erhebliche, jahrelang unbemerkte Überprovisionierung offenlegt — und dann die Abbildung dieser Realität auf eine kleinere Menge geschäftlich bedeutsamer Rollen. Der Gewinn in einem Legacy-Modernisierungskontext ist erheblich: Onboarding und Offboarding werden zu schnellen, risikoarmen Operationen statt manuellen, fehleranfälligen, und der resultierende Audit-Trail unterstützt direkt die regulatorischen Compliance-Verpflichtungen, denen Legacy-Systeme in regulierten Branchen häufig nicht gerecht werden.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken, um diese Lösung im Kontext eines Legacy-Systems umzusetzen.

- Inventarisieren Sie alle bestehenden Zugriffskontrollmechanismen im Legacy-System, um aktuelle Autorisierungsmuster zu verstehen
- Definieren Sie eine klare Rollenhierarchie basierend auf Geschäftsfunktionen und dem Prinzip der geringsten Rechte
- Ordnen Sie bestehende Nutzerberechtigungen den neuen Rollendefinitionen zu und identifizieren Sie überprovisionierte Konten
- Führen Sie einen zentralisierten Autorisierungsdienst oder ein Modul ein, das alle Anwendungskomponenten für Zugriffsentscheidungen nutzen
- Ersetzen Sie verstreute inline Berechtigungsprüfungen durch konsistente rollenbasierte Wächter
- Implementieren Sie Audit-Logging für alle Zugriffskontrollentscheidungen, um Compliance und forensische Analyse zu unterstützen
- Migrieren Sie Legacy-Servicekonten und gemeinsame Anmeldedaten zu rollenbasierten Identitäten

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie dafür aufgeben.

**Vorteile:**
- Vereinfacht Berechtigungsverwaltung, indem Zugriffsrechte in geschäftlich bedeutsame Rollen gruppiert werden
- Reduziert das Risiko von Privilege Escalation durch konsistente Durchsetzung
- Unterstützt regulatorische Compliance durch klare, auditierbare Zugriffskontrollrichtlinien
- Macht Onboarding und Offboarding effizienter und weniger fehleranfällig

**Kosten und Risiken:**
- RBAC nachträglich in Legacy-Systeme mit Ad-hoc-Autorisierungslogik einzubauen erfordert erhebliches Refactoring
- Rollenexplosion kann auftreten, wenn Rollen zu granular sind, was das System schwerer zu verwalten macht
- Der Übergang von individuellen Berechtigungen zu Rollen kann Nutzer-Arbeitsabläufe vorübergehend stören
- Legacy-Integrationen, die gemeinsame Anmeldedaten nutzen, könnten sich der Migration zu rollenbasierten Modellen widersetzen

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Das Legacy-Dokumentenverwaltungssystem einer Regierungsbehörde nutzte ein flaches Berechtigungsmodell, bei dem jeder Nutzer individuell zugewiesene Zugriffsrechte auf spezifische Ordner und Dokumenttypen hatte. Mit über 2.000 Nutzern war die Verwaltung von Berechtigungen zu einer Vollzeitaufgabe für zwei Administratoren geworden. Das Team definierte 12 Rollen basierend auf Abteilungsfunktionen und migrierte alle Nutzer über drei Monate zu rollenbasierten Zuweisungen. Die Berechtigungsverwaltungszeit fiel um 80 %, und ein Audit offenbarte, dass 340 Nutzer zuvor übermäßige Zugriffsrechte gehalten hatten, die das neue Rollenmodell korrekt einschränkte.
