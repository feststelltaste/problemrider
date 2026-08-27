---
title: Abnahmetests durch Nutzer
description: Bestätigung der Erfüllung von Anforderungen durch formale
  Abnahmetests mit Nutzern.
category:
- Testing
- Requirements
problems:
- misaligned-deliverables
- customer-dissatisfaction
- requirements-ambiguity
- insufficient-testing
- implementation-rework
- stakeholder-confidence-loss
- negative-user-feedback
- quality-blind-spots
- reduced-feature-quality
layout: solution
lang: de
en_slug: user-acceptance-tests
related_solutions:
- slug: usability-tests
  similarity: 0.75
- slug: acceptance-tests
  similarity: 0.75
- slug: user-stories
  similarity: 0.75
- slug: prototypes
  similarity: 0.7
- slug: requirements-analysis
  similarity: 0.7
- slug: user-centered-design
  similarity: 0.7
---

## Description

Abnahmetests durch Nutzer sind eine formale Validierungsstufe, in der die tatsächlichen Nutzer eines Systems — nicht Entwickler, nicht QA-Ingenieure — verifizieren, dass ein Ersatz oder eine neue Fähigkeit die echten Geschäftsworkflows, von denen sie abhängen, korrekt unterstützt, unter Nutzung von Abnahmekriterien, die kollaborativ vereinbart wurden, bevor die Entwicklung begann. Es unterscheidet sich von automatisiertem und entwicklergeführtem Testing darin, was es erfassen kann: interne Korrektheit und Unit-Ebenen-Verhalten werden anderswo abgedeckt, aber nur die Personen, die die Arbeit täglich verrichten, können erkennen, wenn eine technisch korrekte Implementierung dennoch nicht dem entspricht, wie die Arbeit tatsächlich erledigt wird. Diese Unterscheidung ist entscheidend in der Legacy-Modernisierung, wo Ersatzsysteme gegen dokumentierte Anforderungen gebaut werden, die unweigerlich stillschweigendes Wissen verpassen, das in Jahren undokumentierter Workarounds und gewohnheitsmäßiger Nutzungsmuster eingebettet ist, die es nie in irgendeine Spezifikation schafften. Die Strukturierung von UAT um vollständige durchgängige Geschäftsworkflows, ausgeführt gegen produktionsähnliche Daten, bringt genau diese Lücken zutage — eine Legacy-Fähigkeit, auf die still vertraut wurde und die die Designer des neuen Systems nie zu replizieren wussten —, während noch Zeit ist, bevor das Legacy-System außer Betrieb genommen wird und Rollback teuer oder unmöglich wird. Da UAT ganz am Ende der Lieferpipeline sitzt, können Befunde in dieser Stufe zeitplan-bedrohend sein, was explizite Freigabekriterien und ausreichende Vorlaufzeit vor dem Go-Live essenziell statt optional macht.

## How to Apply ◆

> In der Legacy-Modernisierung dienen Abnahmetests durch Nutzer als letztes Tor vor der Außerbetriebnahme von Legacy-Komponenten und stellen sicher, dass der Ersatz tatsächlich für die Personen funktioniert, die davon abhängen.

- Definieren Sie Abnahmekriterien kollaborativ mit Nutzern, bevor die Entwicklung beginnt, unter Nutzung konkreter Szenarien aus ihrer täglichen Arbeit mit dem Legacy-System.
- Strukturieren Sie UAT um vollständige Geschäftsworkflows statt um einzelne Features — Nutzer müssen verifizieren, dass durchgängige Prozesse funktionieren, nicht nur isolierte Funktionen.
- Stellen Sie Nutzern während UAT produktionsähnliche Daten zur Verfügung, idealerweise anonymisierte Kopien echter Daten aus dem Legacy-System, um sicherzustellen, dass Tests tatsächliche Nutzungsbedingungen widerspiegeln.
- Planen Sie UAT mit genug Zeit, damit Nutzer gründliches Testen durchführen können und das Entwicklungsteam Befunde vor Go-Live-Fristen adressieren kann.
- Verfolgen Sie UAT-Defekte separat von anderen Defekttypen und verlangen Sie, dass alle kritischen UAT-Befunde behoben werden, bevor die Genehmigung zur Außerbetriebnahme des Legacy-Systems erfolgt.
- Beziehen Sie Regressions-UAT-Zyklen nach bedeutenden Änderungen ein, um zu verifizieren, dass Fixes keine neuen Probleme in zuvor akzeptierter Funktionalität einführen.

## Tradeoffs ⇄

> UAT bietet definitive Validierung, dass der Ersatz Nutzerbedürfnisse erfüllt, erfordert aber erhebliche Koordination und Nutzerverpflichtung.

**Vorteile:**

- Bietet formale Bestätigung, dass das Ersatzsystem Geschäftsanforderungen erfüllt, bevor das Legacy-System ausgemustert wird, was das Go-Live-Risiko reduziert.
- Erfasst Probleme, die automatisierte Tests und Entwicklertesting übersehen, weil sie echtes Domänenwissen zur Identifikation erfordern.
- Schafft Rechenschaftspflicht für die Freigabe und stellt sicher, dass Nutzer den Ersatz explizit genehmigt haben, bevor das Legacy-System ausgemustert wird.
- Baut Nutzer-Ownership des Ersatzsystems auf, indem sie in den Qualitätssicherungsprozess einbezogen werden.

**Kosten und Risiken:**

- UAT erfordert erhebliche Nutzerzeit, was mit ihren regulären Pflichten kollidieren und zu oberflächlichem Testing unter Zeitdruck führen kann.
- Wenn UAT als Formalität statt genuines Testing behandelt wird, werden kritische Probleme in die Produktion entweichen.
- Späte UAT-Entdeckungen können Migrationszeitpläne entgleisen lassen, wenn sie grundlegende Designprobleme offenbaren, die umfangreiche Nacharbeit erfordern.
- Nutzer könnten UAT als Gelegenheit nutzen, neue Features anzufragen, statt vereinbarte Anforderungen zu validieren, was zu Scope Creep führt.

## How It Could Be

> Das folgende Szenario demonstriert die Bedeutung strukturierter UAT beim Ersatz von Legacy-Systemen.

Ein Großhandelsunternehmen migrierte von einem Legacy-Auftragsverwaltungssystem zu einer modernen Plattform. Das Entwicklungsteam hatte alle automatisierten Tests und interne QA bestanden, aber während UAT entdeckten Auftragserfassungssachbearbeiter, dass das Ersatzsystem geteilte Lieferungen nicht auf dieselbe Weise handhaben konnte wie das Legacy-System — das Legacy-System erlaubte Sachbearbeitern, einen Auftrag während der Erfassung über Lager hinweg zu teilen, während das neue System eine Teilung erst nach Einreichung des Auftrags erforderte. Dieser Workflow-Unterschied hätte jedem Multi-Lager-Auftrag einen zusätzlichen Schritt hinzugefügt und 30 % der täglichen Transaktionen betroffen. Weil UAT drei Wochen vor dem geplanten Go-Live angesetzt war, hatte das Team Zeit, die Teilung-während-der-Erfassung-Fähigkeit zu implementieren und einen Regressions-UAT-Zyklus durchzuführen. Ohne strukturierte UAT wäre dieses Problem am ersten Tag der Produktionsnutzung entdeckt worden, was potenziell einen Rollback zum Legacy-System erfordert hätte.
