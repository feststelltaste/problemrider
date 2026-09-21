---
title: Threat Modeling
description: Durchführung systematischer Analyse von Bedrohungen,
  Angreifern und Gegenmaßnahmen.
category:
- Security
- Architecture
problems:
- implementation-starts-without-design
- quality-blind-spots
- architectural-mismatch
- authentication-bypass-vulnerabilities
- authorization-flaws
- system-integration-blindness
- stagnant-architecture
layout: solution
lang: de
en_slug: threat-modeling
related_solutions:
- slug: security-architecture-analysis
  similarity: 0.85
- slug: risk-analysis
  similarity: 0.8
- slug: security-frameworks
  similarity: 0.8
- slug: security-certification
  similarity: 0.8
- slug: secure-software-development
  similarity: 0.8
- slug: threat-intelligence
  similarity: 0.8
---

## Description

Threat Modeling ist eine strukturierte analytische Übung, die die Komponenten, Datenflüsse und Vertrauensgrenzen eines Systems kartiert und dann systematisch fragt, was an jedem Punkt schiefgehen könnte — wer es angreifen könnte, wie, und mit welcher Konsequenz. Methodologien wie STRIDE oder PASTA geben diesem Prozess eine wiederholbare Checkliste, statt sich auf die Risiken zu verlassen, die zufällig denjenigen einfallen, die im Raum sind, was zählt, weil unassistierte Intuition dazu neigt, sich auf vertraute, kürzlich diskutierte Bedrohungen zu konzentrieren und den Rest zu übersehen. In Legacy-Systemen ist Threat Modeling besonders wertvoll, genau weil die ursprüngliche Design-Begründung üblicherweise verschwunden ist: Annahmen über Netzwerkvertrauen, Nutzerverhalten oder Deployment-Topologie, die vernünftig waren, als das System gebaut wurde, sind oft still falsch geworden, während sich die umgebende Umgebung entwickelte, und niemand, der derzeit im Team ist, entschied, das resultierende Risiko zu akzeptieren — es sammelte sich einfach unbemerkt an. Die Produktion eines expliziten Diagramms und einer Bedrohungsliste zwingt diese historischen Annahmen ins Offene, wo sie gegen die aktuelle Bedrohungslandschaft bewertet werden können, statt standardmäßig geerbt zu werden. Die Ausgabe gibt der Sicherheitsinvestition auch eine rationale Grundlage in einer Umgebung, in der Sanierungsressourcen begrenzt sind und Legacy-Architektur nicht immer vollständig neu gestaltet werden kann, was Teams erlaubt, Aufwand auf die risikoreichsten Expositionen zu lenken, statt ihn gleichmäßig über ein System zu verteilen, das niemand mehr vollständig versteht.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken, um diese Lösung im Kontext eines Legacy-Systems umzusetzen.

- Erstellen Sie Datenflussdiagramme des Legacy-Systems, die alle Eintrittspunkte, Datenspeicher und Vertrauensgrenzen identifizieren
- Wenden Sie eine strukturierte Methodologie wie STRIDE oder PASTA an, um systematisch Bedrohungen an jeder Komponente zu identifizieren
- Identifizieren Sie potenzielle Angreifer, ihre Motivationen und für das System relevante Fähigkeiten
- Bewerten Sie identifizierte Bedrohungen nach Risikostufe unter Berücksichtigung sowohl der Wahrscheinlichkeit als auch der Geschäftsauswirkung
- Definieren Sie Gegenmaßnahmen für jede Bedrohung und ordnen Sie sie bestehenden oder geplanten Sicherheitskontrollen zu
- Aktualisieren Sie Bedrohungsmodelle, wenn sich die Systemarchitektur ändert oder neue Bedrohungsinformationen verfügbar werden
- Beziehen Sie sowohl Sicherheitsspezialisten als auch Entwickler mit tiefem Legacy-Systemwissen in den Modellierungsprozess ein

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie dafür aufgeben.

**Vorteile:**
- Bietet strukturierte Identifikation von Sicherheitsrisiken, die Ad-hoc-Ansätze übersehen
- Fokussiert Sicherheitsinvestition auf die wirkungsvollsten Bedrohungen statt Aufwand gleichmäßig zu verteilen
- Schafft gemeinsames Verständnis von Sicherheitsrisiken zwischen Entwicklungs- und Sicherheitsteams
- Produziert Dokumentation, die Sicherheitsentscheidungsfindung und Compliance-Anforderungen unterstützt

**Kosten und Risiken:**
- Threat Modeling erfordert erhebliche Zeitinvestition von erfahrenen Praktikern
- Legacy-Systeme mit schlechter Dokumentation machen genaues Threat Modeling schwierig
- Modelle können schnell veraltet werden, wenn sie nicht zusammen mit Systemänderungen gepflegt werden
- Unvollständige Bedrohungsmodelle können falsches Vertrauen über Sicherheitsabdeckung schaffen
- Analyseparalyse kann auftreten, wenn Threat Modeling zu detailliert oder akademisch wird

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Eine Bank führte ihr erstes Threat Model für ein Legacy-Überweisungssystem durch, das seit 18 Jahren in Produktion war. Die STRIDE-Analyse der Datenflussdiagramme des Systems offenbarte, dass eine interne API, die für Batch-Verarbeitung genutzt wurde, unauthentifizierte Anfragen von jedem Host im internen Netzwerk akzeptierte — eine Annahme, die 2006 vernünftig war, aber angesichts der aktuellen Bedrohungslandschaft gefährlich ist. Das Threat Model identifizierte auch, dass das Logging des Systems unzureichend war, um Transaktionsmanipulation zu erkennen oder zu untersuchen. Diese Befunde trieben gezielte Sicherheitsverbesserungen voran, die die risikoreichsten Bedrohungen adressierten, ohne eine vollständige Systemneuschreibung zu erfordern.
