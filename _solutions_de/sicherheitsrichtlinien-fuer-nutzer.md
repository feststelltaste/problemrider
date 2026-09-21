---
title: Sicherheitsrichtlinien für Nutzer
description: Definition verbindlicher Regeln für die sichere Nutzung von
  Anwendungen.
category:
- Security
- Management
problems:
- password-security-weaknesses
- workaround-culture
- data-protection-risk
- regulatory-compliance-drift
- session-management-issues
- knowledge-gaps
layout: solution
lang: de
en_slug: security-policies-for-users
related_solutions:
- slug: raising-user-awareness
  similarity: 0.8
- slug: security-policies-for-development
  similarity: 0.8
- slug: two-factor-authentication
  similarity: 0.75
- slug: secure-protocols
  similarity: 0.75
- slug: security-monitoring
  similarity: 0.75
- slug: secure-by-default
  similarity: 0.75
---

## Description

Sicherheitsrichtlinien für Nutzer definieren die verbindlichen Regeln, die regeln, wie Menschen erwartungsgemäß sicher mit einer Anwendung interagieren sollen — Passwortanforderungen, akzeptable Nutzung, Datenbehandlung, Fernzugriff —, klar genug formuliert, um durchgesetzt und in Begriffen kommuniziert zu werden, die mit den tatsächlich genutzten Systemen verknüpft sind. In Legacy-Umgebungen drängen Lücken in den Fähigkeiten einer Anwendung selbst Nutzer häufig zu riskanten Workarounds, wie dem Teilen von Anmeldedaten, weil dem System eine Delegationsfunktion fehlt, sodass eine effektive Nutzerrichtlinie mit der Schließung der zugrunde liegenden Lücke gepaart werden muss, statt nur den Workaround zu verbieten. Richtlinien, die wo möglich durch technische Kontrollen durchgesetzt werden, statt rein auf freiwillige Compliance zu setzen, halten besser stand als reine Dokumentation, aber übermäßig belastende Regeln, die ignorieren, wie Menschen tatsächlich arbeiten müssen, neigen dazu, genau die unsicheren Workarounds zu erzeugen, die sie verhindern sollten.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken, um diese Lösung im Kontext eines Legacy-Systems umzusetzen.

- Definieren Sie klare, durchsetzbare Richtlinien für Passwortkomplexität, -rotation und Multi-Faktor-Authentifizierungsnutzung
- Etablieren Sie Richtlinien akzeptabler Nutzung, die Datenbehandlung, Gerätesicherheit und Fernzugriff abdecken
- Kommunizieren Sie Richtlinien in einfacher Sprache mit konkreten Beispielen, relevant für die Anwendungen, mit denen Nutzer interagieren
- Implementieren Sie wo möglich technische Kontrollen, die Richtlinien durchsetzen, statt sich allein auf Nutzer-Compliance zu verlassen
- Erstellen Sie einen Prozess für Richtlinienausnahmeanfragen mit angemessenen Genehmigungsabläufen
- Bieten Sie Schulungsmaterialien, die die Begründung hinter jeder Richtlinienanforderung erklären
- Überprüfen und aktualisieren Sie Nutzerrichtlinien, während sich die Anwendungslandschaft und Bedrohungsumgebung weiterentwickeln

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie dafür aufgeben.

**Vorteile:**
- Setzt klare Erwartungen an sicheres Verhalten, für die Nutzer zur Verantwortung gezogen werden können
- Reduziert Risiko durch nutzerseitige Sicherheitslapsus wie schwache Passwörter und Datenfehlbehandlung
- Unterstützt regulatorische Compliance durch Dokumentation erforderlicher Sicherheitspraktiken
- Bietet ein Framework, um Sicherheitsverstöße konstruktiv zu adressieren

**Kosten und Risiken:**
- Übermäßig belastende Richtlinien führen zu Workarounds, die weniger sicher sein können als das Verhalten, das sie ersetzen
- Nutzer könnten sich Richtlinien widersetzen, die ihre etablierten Arbeitsabläufe erheblich ändern
- Richtliniendurchsetzung in Legacy-Systemen könnte zusätzliches Tooling erfordern, das die Plattform nicht nativ unterstützt
- Compliance-Überwachung für Nutzerverhaltensrichtlinien erfordert Investition in Auditfähigkeiten

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Ein Finanzinstitut entdeckte, dass Nutzer seiner Legacy-Handelsplattform Anmeldedaten teilten, um das Fehlen von Delegationsfeatures im System zu umgehen. Das Sicherheitsteam erstellte eine Nutzersicherheitsrichtlinie, die das Teilen von Anmeldedaten verbot, und arbeitete gleichzeitig mit dem Entwicklungsteam zusammen, um ein Delegationsfeature zur Legacy-Anwendung hinzuzufügen. Die Richtlinie enthielt klare Konsequenzen für Verstöße, bot aber auch eine legitime Alternative. Vorfälle des Teilens von Anmeldedaten fielen innerhalb von zwei Monaten um 90 %, und das Delegationsfeature wurde zu einer der am meisten genutzten Ergänzungen der Plattform.
