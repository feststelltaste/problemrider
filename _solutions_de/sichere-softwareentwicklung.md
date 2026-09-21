---
title: Sichere Softwareentwicklung
description: Etablierung von Sicherheit als integralem Bestandteil des
  Entwicklungsprozesses.
category:
- Security
- Process
problems:
- insufficient-testing
- inadequate-code-reviews
- implementation-starts-without-design
- process-design-flaws
- high-bug-introduction-rate
- quality-compromises
- inconsistent-quality
layout: solution
lang: de
en_slug: secure-software-development
related_solutions:
- slug: secure-coding-guidelines
  similarity: 0.85
- slug: security-tests
  similarity: 0.85
- slug: security-policies-for-development
  similarity: 0.85
- slug: security-certification
  similarity: 0.8
- slug: security-training
  similarity: 0.8
- slug: security-frameworks
  similarity: 0.8
---

## Description

Sichere Softwareentwicklung ist die Praxis, Sicherheitsaktivitäten — Threat Modeling, sicherheitsfokussierte Anforderungen, automatisiertes Scanning, sicherheitsbewusstes Code-Review — in jede Phase des Entwicklungslebenszyklus einzubetten, statt Sicherheit als abschließendes Gate zu behandeln, das kurz vor der Veröffentlichung angewendet wird. Der Mechanismus funktioniert, indem Sicherheitsentscheidungen an den Punkt im Prozess verschoben werden, wo sie am günstigsten zu treffen sind: eine während des Designs identifizierte Bedrohung kostet ein Gespräch, dieselbe in einem Vor-Veröffentlichungs-Audit gefundene Bedrohung kostet eine Neugestaltung, und eine nach einem Vorfall gefundene kostet einen Ausfall und möglicherweise regulatorische Exposition. Legacy-Systeme institutionalisieren üblicherweise das gegenteilige Muster, bei dem Sicherheit als separate, spät angesiedelte Audit-Funktion existiert, getrennt von der täglichen Entwicklung, weil diese Regelung dazu passte, wie die Organisation strukturiert war, als das System gebaut wurde, und einfach fortbestanden hat. Dies umzukehren erfordert mehr als das Hinzufügen von Werkzeugen; es erfordert, Sicherheit zu einem Teil dessen zu machen, was „fertig" für eine User Story bedeutet, Entwicklungsteams die Mittel zu geben, ihre eigene Arbeit durch CI-integriertes Scanning zu prüfen, und Feedback-Schleifen wie sicherheitsfokussierte Retrospektiven zu schaffen, sodass Lektionen aus vergangenen Vorfällen künftiges Verhalten ändern, statt mit dem Vorfallbericht abgelegt zu werden. Für die Legacy-Modernisierung speziell ist diese Lösung wertvoll, weil sie verhindert, dass der in die Verbesserung eines Teils des Systems investierte Aufwand durch neu eingeführte Schwachstellen anderswo untergraben wird, und sie verwandelt Sicherheit von einem wiederkehrenden Reibungspunkt zwischen Sicherheits- und Delivery-Teams in eine gemeinsame, laufende Verantwortung.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken, um diese Lösung im Kontext eines Legacy-Systems umzusetzen.

- Integrieren Sie Sicherheitsaktivitäten in jede Phase des Entwicklungslebenszyklus, von Anforderungen bis Bereitstellung
- Beziehen Sie Threat Modeling während der Designphase für alle bedeutenden Änderungen am Legacy-System ein
- Fügen Sie sicherheitsfokussierte Abnahmekriterien zu User Stories und Feature-Anforderungen hinzu
- Implementieren Sie automatisiertes Sicherheitstesting in der CI/CD-Pipeline, einschließlich SAST, DAST und Abhängigkeitsscanning
- Benennen Sie Sicherheits-Champions innerhalb von Entwicklungsteams, um Sicherheitsbewusstsein und -praktiken voranzutreiben
- Führen Sie sicherheitsfokussierte Retrospektiven durch, um aus Vorfällen und Beinahe-Unfällen zu lernen
- Etablieren Sie Sicherheits-Gates im Veröffentlichungsprozess, die vor der Bereitstellung bestanden werden müssen

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie dafür aufgeben.

**Vorteile:**
- Erfasst Sicherheitsprobleme früh, wenn sie günstiger zu beheben sind
- Verschiebt Sicherheit von einer Gatekeeper-Funktion zu einer gemeinsamen Teamverantwortung
- Reduziert die Anzahl der Schwachstellen, die Produktion erreichen
- Schafft einen wiederholbaren Prozess, der über Teams und Projekte hinweg skaliert

**Kosten und Risiken:**
- Das Hinzufügen von Sicherheitsaktivitäten zum Entwicklungsprozess erhöht anfangs die Zykluszeit
- Erfordert Investition in Tooling, Schulung und potenziell dediziertes Sicherheitspersonal
- Teams könnten Sicherheitsaktivitäten als Checkbox-Übungen ohne echtes Engagement behandeln
- Die Balance zwischen Sicherheitsrigorosität und Liefergeschwindigkeit erfordert laufende Kalibrierung

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Ein Finanzdienstleistungsunternehmen hatte Sicherheit historisch als abschließendes Audit behandelt, durchgeführt von einem externen Team vor jeder Veröffentlichung, was zu kostspieligen spät angesiedelten Befunden und verzögerten Veröffentlichungen führte. Es wechselte zu einem sicheren Entwicklungslebenszyklus, indem es Threat Modeling in die Sprint-Planung einbettete, automatisierte SAST-Scans zu Pull-Request-Prüfungen hinzufügte und zwei Entwickler pro Team als Sicherheits-Champions schulte. Innerhalb von sechs Monaten fiel die Anzahl der Sicherheitsbefunde in der abschließenden Audit-Phase um 70 %, und die durchschnittliche Veröffentlichungszykluszeit sank um zwei Wochen, weil weniger spät angesiedelte Nacharbeitszyklen nötig waren.
