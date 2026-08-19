---
title: Code-Coverage-Analyse
description: Messung des Anteils an Code, der von Tests abgedeckt wird.
category:
- Testing
problems:
- poor-test-coverage
- legacy-code-without-tests
- insufficient-testing
- fear-of-breaking-changes
- regression-bugs
- test-debt
- quality-blind-spots
layout: solution
lang: de
en_slug: code-coverage-analysis
related_solutions:
- slug: test-coverage-strategy
  similarity: 0.8
- slug: static-analysis-and-linting
  similarity: 0.8
- slug: code-metrics
  similarity: 0.8
- slug: regression-testing
  similarity: 0.75
- slug: code-quality-gates
  similarity: 0.75
- slug: quality-ratchet
  similarity: 0.7
---

## Description

Code-Coverage-Analyse misst, welcher Anteil der Zeilen, Zweige oder Pfade einer Codebasis tatsächlich von einer automatisierten Test-Suite ausgeübt wird, und verwandelt eine sonst unsichtbare Eigenschaft des Systems — wie viel davon tatsächlich vor Regression geschützt ist — in eine quantifizierbare, verfolgbare Metrik. Coverage-Werkzeuge werden in die Build-Pipeline integriert, sodass die Metrik bei jeder Änderung gemessen und als Trend statt als einmalige Momentaufnahme verfolgt werden kann. In Legacy-Systemen, die häufig wenig bis keine automatisierte Testabdeckung von Anfang an haben, macht diese Messung es möglich, präzise zu identifizieren, wo die risikoreichsten blinden Flecken sind, besonders durch das Kreuzreferenzieren von Coverage-Daten mit Änderungshäufigkeit: Code, der sowohl häufig geändert als auch ungetestet ist, ist dort, wo Regressionen am wahrscheinlichsten entstehen. Statt eine einheitliche Coverage-Prozentzahl über eine gesamte Legacy-Codebasis anzustreben, was selten ein realistisches oder wertvolles Ziel ist, unterstützt Coverage-Analyse eine gezielte Strategie, begrenzten Testaufwand auf die spezifischen Module zu lenken, wo er das meiste Risiko verringert. Eine Coverage-Ratsche — eine Regel, die verbietet, dass die Gesamtprozentzahl sinkt — schützt außerdem bereits erreichte Gewinne, indem sichergestellt wird, dass neuer Code das aufgebaute Sicherheitsnetz nicht still erodiert. Die Metrik hat jedoch einen bekannten Fehlermodus: Eine hohe Coverage-Zahl kann falsches Vertrauen schaffen, wenn sie nur widerspiegelt, dass Zeilen ausgeführt werden, statt dass ihr Verhalten und ihre Randfälle sinnvoll geprüft werden, sodass Coverage als Indikator dafür gelesen werden sollte, wo Tests fehlen, nicht als Beweis, dass bestehende Tests angemessen sind.

## How to Apply ◆

> In Legacy-Systemen offenbart Code-Coverage-Analyse, welche Teile der Codebasis von Tests geschützt sind und welche blinde Flecken sind, wo Änderungen das höchste Risiko tragen.

- Integrieren Sie ein Coverage-Analyse-Werkzeug (JaCoCo, Istanbul, coverage.py) in die CI-Pipeline, um Coverage bei jedem Build zu messen und Trends über die Zeit zu verfolgen.
- Nutzen Sie Coverage-Daten, um die riskantesten Teile der Legacy-Codebasis zu identifizieren — Module mit hoher Änderungshäufigkeit, aber geringer Testabdeckung haben höchste Priorität für Testinvestition.
- Setzen Sie realistische Coverage-Ziele, die schrittweise steigen, statt sofortige hohe Coverage für eine Legacy-Codebasis zu verlangen, die möglicherweise nahe null beginnt.
- Setzen Sie eine Coverage-Ratsche durch, die verhindert, dass Coverage sinkt — neue Änderungen müssen die Gesamtcoverage-Prozentzahl beibehalten oder verbessern.
- Unterscheiden Sie zwischen Zeilencoverage, Zweigcoverage und Mutation-Testing-Ergebnissen — reine Zeilencoverage kann falsches Vertrauen schaffen, wenn bedingte Logik nicht vollständig getestet ist.
- Nutzen Sie Coverage-Berichte während Code-Reviews, um zu verifizieren, dass neuer Code und geänderter Legacy-Code angemessene Testabdeckung beinhalten.
- Fokussieren Sie Coverage-Verbesserungsbemühungen auf geschäftskritische Pfade und häufig geänderten Code, statt eine einheitliche Coverage-Prozentzahl über die gesamte Codebasis anzustreben.

## Tradeoffs ⇄

> Coverage-Analyse identifiziert Testlücken, kann aber zu einer irreführenden Metrik werden, wenn sie als Ziel an sich statt als Werkzeug für Risikomanagement verfolgt wird.

**Vorteile:**

- Macht Testlücken sichtbar, was es dem Team erlaubt, Testinvestition dort zu priorisieren, wo sie den größten Risikoreduktionseffekt hat.
- Bietet eine objektive Metrik zur Verfolgung von Testverbesserung über die Zeit während der Legacy-Modernisierung.
- Hilft, toten Code zu identifizieren — Code mit null Coverage, der auch nie in Produktion erreicht wird, könnte sicher entfernbar sein.
- Verhindert Coverage-Regression, indem das Team alarmiert wird, wenn Änderungen den Anteil getesteten Codes verringern.

**Kosten und Risiken:**

- Hohe Coverage-Zahlen können falsches Vertrauen schaffen — 80 % Zeilencoverage bedeutet nicht, dass 80 % des Verhaltens getestet sind, wenn Randfälle und Fehlerpfade unabgedeckt bleiben.
- Coverage als Ziel statt als Werkzeug zu verfolgen kann zu geringwertigen Tests führen, die die Coverage-Zahl erhöhen, ohne Risiko sinnvoll zu verringern.
- Coverage-Analyse fügt dem Build-Prozess Zeit hinzu, was in Legacy-Systemen mit bereits langen Build-Zeiten unwillkommen sein könnte.
- Der Fokus auf Coverage-Metriken kann Aufmerksamkeit von Testqualität ablenken — einige gut designte Tests bieten oft mehr Schutz als viele oberflächliche.

## How It Could Be

> Das folgende Szenario demonstriert, wie Coverage-Analyse Testinvestition in einer Legacy-Codebasis leitet.

Das Legacy-Zahlungsverarbeitungssystem eines Fintech-Unternehmens hatte 12 % Gesamttestabdeckung. Statt ein pauschales 80-%-Coverage-Ziel zu setzen, nutzte das Team Coverage-Analyse kombiniert mit Änderungshäufigkeitsdaten, um die 30 Klassen zu identifizieren, die sowohl häufig geändert als auch null Testabdeckung hatten. Diese Klassen machten 60 % der Produktionsdefekte aus. Durch Fokussierung der Testinvestition zuerst auf diese Hochrisikoklassen erhöhte das Team die Gesamtcoverage auf 25 %, deckte aber 85 % des häufig geänderten Codes ab. Über das folgende Jahr sanken Produktionsdefekte in den anvisierten Klassen um 70 %, was die Strategie validierte, Coverage-Daten für risikobasierte Testpriorisierung statt einheitlicher Coverage-Ziele zu nutzen.
