---
title: Sicheres Session-Management
description: Verwaltung von Sessions basierend auf zufälligen, zeitlich
  begrenzten IDs.
category:
- Security
problems:
- session-management-issues
- authentication-bypass-vulnerabilities
- authorization-flaws
- cross-site-scripting-vulnerabilities
- data-protection-risk
- password-security-weaknesses
layout: solution
lang: de
en_slug: secure-session-management
related_solutions:
- slug: secure-protocols
  similarity: 0.75
- slug: cryptographic-methods
  similarity: 0.75
- slug: secure-by-default
  similarity: 0.7
- slug: secure-programming-interfaces
  similarity: 0.7
- slug: secure-configuration
  similarity: 0.7
- slug: secure-coding-guidelines
  similarity: 0.7
---

## Description

Sicheres Session-Management ist die Praxis, einen authentifizierten Nutzer über mehrere Anfragen hinweg mittels Session-Tokens zu identifizieren, die unvorhersehbar, zeitlich begrenzt und ordnungsgemäß invalidiert werden, statt sich auf Identifikatoren zu verlassen, die erratbar, langlebig oder an Orten exponiert sind, an denen ein Angreifer sie beobachten kann. Der zugrunde liegende Mechanismus ruht auf einer Handvoll zusammenwirkender Eigenschaften: Session-Identifikatoren müssen mit einer kryptografisch sicheren Zufallsquelle generiert werden, sodass sie nicht vorhergesagt oder aufgezählt werden können, Sessions müssen nach einer begrenzten Periode der Inaktivität und einer absoluten Höchstlebensdauer ablaufen, Identifikatoren müssen in dem Moment neu generiert werden, in dem sich die Privilegienstufe eines Nutzers ändert (am wichtigsten beim Login), um Session Fixation zu vereiteln, und Session-Zustand sollte serverseitig gespeichert werden, wobei der Client nur eine opake Referenz hält. Legacy-Anwendungen verletzen häufig mehrere dieser Eigenschaften gleichzeitig, weil sie gebaut wurden, bevor Session Hijacking und Fixation als Angriffsklassen gut verstanden waren — sequenzielle Integer-IDs, in URLs eingebettete Session-Tokens oder Session-Lebensdauern, gemessen in Tagen statt Minuten, sind häufige Befunde in vor mehr als einem Jahrzehnt gebauten Systemen. Die Modernisierung der Session-Behandlung zählt unverhältnismäßig für Legacy-Systeme, weil ein kompromittiertes Session-Token einem Angreifer denselben Zugang wie dem legitimen Nutzer gewährt, ohne Authentifizierung überhaupt brechen zu müssen, was es zu einem der direkteren Pfade zur Kontoübernahme in älteren Codebasen macht, die diese Schwächen noch tragen. Da Session-Logik üblicherweise durch viele Teile einer Anwendung gefädelt ist, ist ihre Korrektur selten ein lokalisierter Fix, was sie zu einem echten Modernisierungsaufwand macht statt einer einfachen Konfigurationsänderung.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken, um diese Lösung im Kontext eines Legacy-Systems umzusetzen.

- Ersetzen Sie jegliche vorhersagbaren oder sequenziellen Session-Identifikatoren durch kryptografisch zufällige Tokens
- Implementieren Sie Session-Timeouts mit sowohl Leerlauf- als auch absoluten Ablauflimits
- Regenerieren Sie Session-Identifikatoren nach der Authentifizierung, um Session-Fixation-Angriffe zu verhindern
- Speichern Sie Session-Daten serverseitig statt in clientseitigen Cookies oder Local Storage
- Setzen Sie sichere Cookie-Attribute, einschließlich HttpOnly-, Secure- und SameSite-Flags
- Implementieren Sie Session-Invalidierung beim Logout und bieten Sie Mechanismen zum Widerruf aktiver Sessions
- Fügen Sie Überwachung für anomales Session-Verhalten hinzu, wie gleichzeitige Sessions von unterschiedlichen Standorten

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie dafür aufgeben.

**Vorteile:**
- Verhindert Session Hijacking, Fixation und Replay-Angriffe
- Begrenzt das Schadensfenster kompromittierter Sessions durch zeitbasierten Ablauf
- Unterstützt Compliance-Anforderungen für Authentifizierung und Zugriffskontrolle
- Ermöglicht zentralisiertes Session-Management und -Überwachung

**Kosten und Risiken:**
- Kürzere Session-Timeouts können Nutzer frustrieren, besonders in Legacy-Anwendungen mit langen Arbeitsabläufen
- Serverseitige Session-Speicherung erfordert Infrastruktur für Session-Zustandsverwaltung im großen Maßstab
- Session-Migration während Bereitstellungen erfordert sorgfältige Handhabung, um Nutzerstörung zu vermeiden
- Legacy-Anwendungen mit benutzerdefinierter Session-Behandlung könnten erhebliches Refactoring erfordern

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Ein 2008 gebautes Gesundheitsportal nutzte sequenzielle Integer-Session-IDs, gespeichert in URL-Parametern, was Session Hijacking trivial machte. Das Team migrierte zu kryptografisch zufälligen Session-Tokens, gespeichert in HttpOnly-Cookies mit 30-minütigen Leerlauf-Timeouts. Sie implementierten außerdem Session-Regeneration nach dem Login und fügten Logging für die Erkennung gleichzeitiger Sessions hinzu. Die Migration erforderte die Aktualisierung von 15 Legacy-Modulen, die Session-IDs in URLs übergaben, beseitigte aber alle sessionbezogenen Befunde in der nachfolgenden Sicherheitsbewertung.
