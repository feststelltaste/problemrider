---
title: Unterstützung assistiver Technologien
description: Sicherstellung der Nutzbarkeit assistiver Technologien.
category:
- Requirements
- Code
quality_tactics_url: https://qualitytactics.de/en/usability/assistive-technology-support/
problems:
- poor-user-experience-ux-design
- user-frustration
- user-confusion
- negative-user-feedback
- customer-dissatisfaction
- competitive-disadvantage
- regulatory-compliance-drift
- feature-gaps
layout: solution
lang: de
en_slug: assistive-technology-support
related_solutions:
- slug: accessibility-concept
  similarity: 0.85
- slug: consistent-user-interface
  similarity: 0.75
- slug: intuitive-navigation
  similarity: 0.75
- slug: user-centered-design
  similarity: 0.75
- slug: keyboard-support
  similarity: 0.75
- slug: a-b-testing
  similarity: 0.7
---

## Description

Unterstützung assistiver Technologien stellt sicher, dass eine Legacy-Schnittstelle tatsächlich mit den Werkzeugen funktioniert — Screenreader, Vergrößerungssoftware, alternative Eingabegeräte —, auf die viele Nutzer angewiesen sind, um sie überhaupt zu bedienen, etwas, mit dem die meisten Legacy-UIs nie im Sinn gebaut wurden. Weil diese Systeme typischerweise auf rein visuellen Hinweisen und nicht-semantischem Markup beruhen, bedeutet Nachrüstung von Unterstützung, gegen Barrierefreiheitsstandards wie WCAG zu auditieren, generische Elemente durch ordentliche semantische und ARIA-annotierte Steuerelemente zu ersetzen und mit echten assistiven Werkzeugen zu testen statt nur mit automatisierten Scannern, die nur einen Bruchteil echter Probleme abfangen. Über die direkte Compliance- und rechtliche Exposition hinaus, die dies adressiert, machen dieselben Änderungen — klare Fokusindikatoren, ausreichender Kontrast, aussagekräftige Beschriftungen — die Schnittstelle tendenziell für alle nutzbarer, nicht nur für die Nutzer, auf die die Nachrüstung abzielte.

## How to Apply ◆

> Legacy-Systeme wurden oft ohne jegliche Berücksichtigung assistiver Technologien wie Screenreader, Vergrößerungssoftware oder alternative Eingabegeräte gebaut. Die Nachrüstung von Barrierefreiheitsunterstützung ist essenziell für Compliance und Inklusion.

- Auditieren Sie die bestehende Legacy-Schnittstelle gegen WCAG-2.1-AA-Standards, um Barrierefreiheitslücken zu identifizieren. Nutzen Sie automatisierte Werkzeuge wie axe oder Lighthouse als Ausgangspunkt, aber folgen Sie mit manuellem Testen unter Nutzung echter Screenreader wie NVDA oder JAWS.
- Fügen Sie semantische HTML-Elemente und ARIA-Attribute zu Legacy-Markup hinzu, das auf rein visuellen Hinweisen beruht. Ersetzen Sie `<div>`- und `<span>`-Elemente, die als interaktive Steuerelemente genutzt werden, durch ordentliche `<button>`-, `<input>`- und `<select>`-Elemente.
- Stellen Sie sicher, dass alle Bilder, Icons und Nicht-Text-Inhalte aussagekräftigen alternativen Text haben. In Legacy-Systemen fehlt dekorativen Icons oft vollständig ein Alt-Attribut, was Screenreader dazu bringt, unhilfreiche Dateinamen anzukündigen.
- Testen Sie Farbkontrastverhältnisse in der gesamten Anwendung. Legacy-Systeme nutzen häufig kontrastarme Farbschemata, die für Nutzer mit Sehbehinderungen schwierig sind.
- Implementieren Sie Fokusindikatoren für alle interaktiven Elemente, sodass rein tastaturbasierte Nutzer ihre Position auf der Seite verfolgen können. Viele Legacy-CSS-Resets entfernen standardmäßige Fokus-Umrandungen, ohne Alternativen bereitzustellen.
- Etablieren Sie eine Barrierefreiheits-Checkliste als Teil des Code-Review-Prozesses, sodass neue Änderungen am Legacy-System keine zusätzlichen Hindernisse einführen.

## Tradeoffs ⇄

> Ein Legacy-System barrierefrei zu machen erweitert die Nutzerbasis und stellt rechtliche Compliance sicher, erfordert aber gezielten Aufwand, der mit Feature-Lieferung konkurrieren kann.

**Vorteile:**

- Stellt Compliance mit Barrierefreiheitsvorschriften wie ADA, Section 508 und dem European Accessibility Act sicher und verringert rechtliches Risiko.
- Erweitert die Nutzerbasis um Menschen mit Behinderungen, die zuvor von der Nutzung des Systems ausgeschlossen waren.
- Verbessert die Nutzbarkeit für alle Nutzer, weil Barrierefreiheitsverbesserungen wie klare Beschriftungen, logische Tab-Reihenfolge und konsistente Navigation allen zugutekommen.
- Verringert Kundenunzufriedenheit und negatives Feedback von Nutzern, die auf assistive Technologien angewiesen sind.

**Kosten und Risiken:**

- Die Nachrüstung von Barrierefreiheit in eine Legacy-UI, die vollständig mit nicht-semantischem Markup gebaut wurde, kann zeitaufwendig sein und erhebliche Refaktorierung erfordern.
- Entwickler, die mit Barrierefreiheitsstandards nicht vertraut sind, brauchen Schulung, was Zeit von anderen Prioritäten abzieht.
- Manche Legacy-UI-Muster, wie benutzerdefinierte Drag-and-Drop-Schnittstellen oder komplexe Datengitter, sind inhärent schwierig barrierefrei zu machen und könnten alternative Interaktionsmodi erfordern.
- Laufendes Testen mit assistiven Technologien fügt der QA-Arbeitslast hinzu, weil automatisierte Werkzeuge nur etwa 30 % der Barrierefreiheitsprobleme abfangen.

## How It Could Be

> Legacy-Systeme weisen häufig tiefe Barrierefreiheitsdefizite auf, die dringend werden, wenn sich regulatorische Anforderungen ändern oder die Nutzerbasis wächst.

Das Legacy-Fallmanagementsystem einer Regierungsbehörde wird während eines Barrierefreiheitsaudits markiert, weil keines seiner Formulare per Screenreader navigierbar ist. Die Formulare nutzen tabellenbasierte Layouts ohne semantische Struktur, und Pflichtfelder werden nur durch Farbe angezeigt. Das Team rüstet jedes Formularmodul während geplanter Wartungssprints schrittweise nach und fügt ordentliche Label-Zuordnungen, ARIA-Rollen und sichtbare Pflichtfeld-Indikatoren hinzu. Innerhalb von sechs Monaten besteht das System ein unabhängiges Barrierefreiheitsaudit, und die Behörde vermeidet mögliche rechtliche Schritte. Mehrere Sachbearbeiter mit Sehbehinderungen, die zuvor auf Kollegen angewiesen waren, um Daten einzugeben, können das System nun unabhängig nutzen.
