---
title: Sicherheitsschulung
description: Sensibilisierung und Weiterbildung von Mitarbeitern zu
  Sicherheitsthemen.
category:
- Security
- Culture
problems:
- knowledge-gaps
- inexperienced-developers
- inadequate-onboarding
- skill-development-gaps
- implicit-knowledge
- inadequate-mentoring-structure
- legacy-skill-shortage
layout: solution
lang: de
en_slug: security-training
related_solutions:
- slug: raising-user-awareness
  similarity: 0.85
- slug: secure-software-development
  similarity: 0.8
- slug: security-community
  similarity: 0.8
- slug: secure-coding-guidelines
  similarity: 0.8
- slug: security-certification
  similarity: 0.8
- slug: security-culture
  similarity: 0.8
---

## Description

Sicherheitsschulung baut die Fähigkeit von Entwicklern und Betreibern auf, Sicherheitsprobleme selbst zu erkennen und zu verhindern, durch rollenbasierte Curricula und praktische Übungen, idealerweise gebaut aus der eigenen Codebasis der Organisation statt generischem, von den tatsächlich genutzten Systemen losgelöstem Material. Legacy-Systeme tragen Sicherheitsbelange, die generische Sicherheitsschulung selten direkt adressiert — veraltete Authentifizierungsmechanismen, veraltete APIs, Muster, die akzeptabel waren, als der Code geschrieben wurde, aber heute als Schwachstellen erkannt sind —, sodass Schulung, die die echten Legacy-Schwachstellen des Teams als Lehrmaterial nutzt, genau die Mustererkennung aufbaut, die gebraucht wird, um ähnliche Probleme während Modernisierungsarbeit zu erfassen. Diese Schulung zu einem verpflichtenden, wiederkehrenden Teil des Onboardings und der laufenden Entwicklung zu machen, statt zu einem einmaligen Ereignis, ist es, was das zugrunde liegende Wissen davon abhält zu verblassen, obwohl der Aufbau und die Pflege dieses Schulungsinhalts selbst eine echte Investition ist, die mit Lieferarbeit um dieselbe Entwicklerzeit konkurriert.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken, um diese Lösung im Kontext eines Legacy-Systems umzusetzen.

- Entwickeln Sie rollenbasierte Schulungscurricula, die sichere Programmierung, Sicherheitsarchitektur und Vorfallreaktion abdecken
- Beziehen Sie praktische Übungen ein, die echte Schwachstellenmuster aus der eigenen Legacy-Codebasis der Organisation nutzen
- Bieten Sie Schulung zu legacy-spezifischen Sicherheitsbelangen wie veralteten Authentifizierungsmechanismen und veralteten APIs
- Machen Sie Sicherheitsschulung zu einem verpflichtenden Teil des Onboardings für alle neuen Entwickler und Betriebsmitarbeiter
- Bieten Sie fortgeschrittene Schulungspfade für Sicherheits-Champions und Teamleiter an
- Verfolgen Sie den Abschluss von Schulungen und messen Sie Wissenserhalt durch periodische Bewertungen
- Aktualisieren Sie Schulungsinhalte regelmäßig, um neue Bedrohungen und gelernte Lektionen aus internen Vorfällen widerzuspiegeln

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie dafür aufgeben.

**Vorteile:**
- Baut interne Sicherheitsexpertise auf, die die Abhängigkeit von externen Beratern reduziert
- Ermächtigt Entwickler, Sicherheitsprobleme während der Entwicklung zu identifizieren und zu verhindern
- Schafft eine gemeinsame Sicherheitswissensbasislinie über die Organisation hinweg
- Verbessert die Wirksamkeit von Code-Reviews und Designdiskussionen für Sicherheitsbelange

**Kosten und Risiken:**
- Schulungsentwicklung und -durchführung erfordert erhebliche Zeit- und Ressourceninvestition
- Wissen verblasst ohne regelmäßige Auffrischung und praktische Anwendung
- Generische Schulungsinhalte adressieren möglicherweise nicht die spezifischen Sicherheitsherausforderungen von Legacy-Systemen
- Schulung konkurriert mit Lieferarbeit um Entwicklerzeit und -aufmerksamkeit

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Eine Beratungsfirma, die mehrere Legacy-Java-Anwendungen betreute, erstellte ein Sicherheitsschulungsprogramm, das echte, in der eigenen Codebasis gefundene Schwachstellen als Lehrmaterial nutzte. Entwickler übten das Identifizieren und Beheben von SQL-Injection, unsicherer Deserialisierung und fehlerhafter Zugriffskontrolle in Sandbox-Umgebungen, die ihre Produktionssysteme spiegelten. Nach Abschluss der Schulung stieg die Rate der Sicherheitsbefunde in Code-Reviews um 40 %, was darauf hinwies, dass Entwickler Probleme erfassten, die sie zuvor übersehen hatten. Die Schulung reduzierte außerdem die durchschnittliche Zeit zur Behebung von Sicherheitsbefunden von zwei Wochen auf drei Tage, während Entwickler sowohl die Schwachstellen als auch die Behebungsmuster besser verstanden.
