---
title: Architektur-Governance
description: Definition und Durchsetzung architektonischer Prinzipien und bewährter
  Praktiken.
category:
- Architecture
- Management
problems:
- stagnant-architecture
- high-coupling-low-cohesion
- architectural-mismatch
- inconsistent-codebase
- high-technical-debt
- technology-stack-fragmentation
- convenience-driven-development
- cargo-culting
- accumulated-decision-debt
- cv-driven-development
- premature-technology-introduction
layout: solution
lang: de
en_slug: architecture-governance
related_solutions:
- slug: architecture-review-board
  similarity: 0.8
- slug: architecture-decision-records
  similarity: 0.75
- slug: architecture-conformity-analysis
  similarity: 0.75
- slug: architecture-documentation
  similarity: 0.75
- slug: architecture-roadmap
  similarity: 0.75
- slug: architecture-reviews
  similarity: 0.75
---

## Description

Architektur-Governance ist die Definition und aktive Durchsetzung einer kleinen Menge architektonischer Prinzipien und Regeln — wie das Verbot direkten Datenbankzugriffs aus Präsentationsschichten oder die Anforderung, dass jegliche Inter-Service-Kommunikation über definierte APIs laufen muss — kodiert wo immer möglich in automatisierte Prüfungen, sodass Verstöße während der Entwicklung erfasst werden statt später entdeckt zu werden. Ihre Abwesenheit ist häufig die Grundursache des strukturellen Verfalls, der in Legacy-Systemen gefunden wird: Ohne jegliche Autorität oder Prozess, der architektonische Entscheidungen regelt, trifft jedes Team und jeder dringende Fix unabhängig voneinander lokal vernünftige Entscheidungen, und über Jahre summieren sich diese unkoordinierten Entscheidungen zu fragmentierten Technologie-Stacks, inkonsistenten Mustern und stetig anwachsenden technischen Schulden, ohne dass eine einzelne Entscheidung für das Ergebnis verantwortlich ist. Die Etablierung von Governance kehrt dies um, indem Teams explizite Leitplanken gegeben werden, innerhalb derer sie weiterhin frei handeln können, statt eine fallweise Genehmigung für jede Entscheidung zu verlangen, was den Prozess leichtgewichtig genug hält, um täglichen Entwicklungsdruck zu überstehen, statt ein bürokratisches Hindernis zu werden, das Teams umgehen. Ein Governance-Prozess, der außerdem ein sichtbares Entscheidungsprotokoll pflegt, verwandelt architektonische Entscheidungsfindung von etwas Undurchsichtigem und Persönlichkeitsabhängigem in etwas Transparentes und für jeden in jedem Team Überprüfbares. Weil sich Governance-Regeln weiterentwickeln müssen, während sich das System und seine Einschränkungen ändern, brauchen sie periodische Überprüfung, um relevant zu bleiben, und weil „temporäre" Fixes und dringende Patches so oft die Quelle architektonischer Schäden in Legacy-Systemen sind, funktioniert der Governance-Prozess nur, wenn er auch auf diese Änderungen angewendet wird, nicht nur auf geplante, bewusste. Das Hauptrisiko ist, dass Governance, die ohne Input der Teams definiert wird, die unter ihr leben müssen, unpraktikabel wird oder als reine Überwachung statt echte Unterstützung für bessere Entscheidungen wahrgenommen wird.

## How to Apply ◆

> In Legacy-Systemen ist die Abwesenheit von Architektur-Governance oft die Grundursache jahrzehntelangen angehäuften strukturellen Verfalls — die Etablierung von Governance bietet die Leitplanken für nachhaltige Modernisierung.

- Definieren Sie eine kleine Menge nicht verhandelbarer architektonischer Prinzipien (z. B. kein direkter Datenbankzugriff aus Präsentationsschichten, jegliche Inter-Service-Kommunikation über definierte APIs) und kommunizieren Sie sie klar an alle Teams.
- Kodieren Sie architektonische Regeln in automatisierten Werkzeugen (Linter, Architektur-Tests, Abhängigkeitsprüfer), sodass Verstöße während der Entwicklung erfasst werden statt in Reviews oder Produktion.
- Etablieren Sie einen leichtgewichtigen Governance-Prozess für architektonische Entscheidungen, der Kontrolle mit Team-Autonomie ausbalanciert — Teams sollten befähigt werden, Entscheidungen innerhalb von Leitplanken zu treffen, statt auf Genehmigung wartend blockiert zu sein.
- Erstellen Sie ein Architektur-Entscheidungsprotokoll, das bedeutende Entscheidungen, ihren Kontext und ihre Begründung aufzeichnet und Governance transparent statt undurchsichtig macht.
- Überprüfen und aktualisieren Sie Governance-Regeln periodisch, um die sich entwickelnde Architektur widerzuspiegeln, und entfernen Sie Regeln, die nicht mehr relevant sind.
- Stellen Sie sicher, dass Governance auf alle Änderungen angewendet wird, einschließlich „temporärer" Fixes und dringender Patches — dies sind oft die Änderungen, die den größten architektonischen Schaden in Legacy-Systemen verursachen.

## Tradeoffs ⇄

> Architektur-Governance verhindert strukturellen Verfall, muss aber ausbalanciert werden, um nicht zu einem bürokratischen Engpass zu werden.

**Vorteile:**

- Verhindert die schrittweise Erosion architektonischer Integrität, die gut strukturierte Systeme über die Zeit in unwartbaren Legacy-Code verwandelt.
- Schafft Konsistenz über Teams hinweg, indem gemeinsame Standards dafür etabliert werden, wie Komponenten strukturiert sein sollten und wie sie interagieren sollten.
- Verringert die Anhäufung technischer Schulden, indem sichergestellt wird, dass neue Entwicklung architektonischen Richtlinien folgt.
- Bietet einen Rahmen zur Bewertung von Technologiewahlen und verhindert unkontrollierte Technologie-Stack-Wucherung.

**Kosten und Risiken:**

- Übermäßig starre Governance kann die Entwicklung verlangsamen und Teams frustrieren, was zu Workarounds führt, die den Governance-Prozess untergraben.
- Governance erfordert architektonische Expertise, die in Organisationen, die auf Legacy-Systempflege fokussiert sind, knapp sein kann.
- Regeln, die ohne Input von Entwicklungsteams definiert werden, können unpraktikabel oder mit tatsächlichen Entwicklungsherausforderungen fehlausgerichtet sein.
- Governance, die sich ausschließlich auf die Verhinderung von Verstößen fokussiert, ohne Anleitung und Unterstützung zu bieten, wird als Überwachung statt als Befähigung wahrgenommen.

## How It Could Be

> Das folgende Szenario zeigt, wie Architektur-Governance weiteren Verfall während der Legacy-Modernisierung verhindert.

Ein Versicherungsunternehmen mit 12 Entwicklungsteams, die eine gemeinsame Legacy-Plattform pflegten, hatte keine Architektur-Governance. Über 15 Jahre hatten Teams fünf verschiedene ORMs, drei Logging-Frameworks, vier Authentifizierungsmechanismen und unzählige Ad-hoc-Integrationsmuster eingeführt. Als das Unternehmen mit der Modernisierung begann, etablierte es ein Architektur-Governance-Board, das drei Kategorien von Entscheidungen definierte: Team-Ebene-Entscheidungen (frei zu treffen), teamübergreifende Entscheidungen (erfordern Peer-Review von betroffenen Teams) und strategische Entscheidungen (erfordern Board-Genehmigung). Sie kodierten außerdem 20 Kern-Architekturregeln als automatisierte Prüfungen in der CI-Pipeline. Innerhalb eines Jahres trat keine neue Technologie-Stack-Fragmentierung auf, und die Anzahl teamübergreifender Integrationsprobleme sank um 60 %. Das Governance-Board traf sich alle zwei Wochen für 30 Minuten, was bewies, dass effektive Governance keine schwere Bürokratie erfordert.
