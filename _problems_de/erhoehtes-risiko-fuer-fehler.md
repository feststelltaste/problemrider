---
title: Erhöhtes Risiko für Fehler
description: Codekomplexität und mangelnde Klarheit machen es wahrscheinlicher, dass
  Entwickler bei Änderungen Defekte einführen.
category:
- Code
related_problems:
- slug: high-bug-introduction-rate
  similarity: 0.75
- slug: increased-bug-count
  similarity: 0.7
- slug: increased-cost-of-development
  similarity: 0.7
- slug: debugging-difficulties
  similarity: 0.65
- slug: lower-code-quality
  similarity: 0.65
- slug: fear-of-change
  similarity: 0.65
solutions:
- contract-testing
- development-workflow-automation
- regression-testing
- functional-tests
- property-based-testing
- value-range-definition
- code-generation
- code-reviews
- code-review-guidelines
- exploratory-testing
- defect-triage-process
layout: problem
lang: de
en_slug: increased-risk-of-bugs
---

## Description

Erhöhtes Risiko für Fehler tritt auf, wenn die Struktur, Komplexität oder Klarheit von Code es wahrscheinlicher macht, dass Entwickler während Entwicklungs- oder Wartungsaktivitäten Defekte einführen. Dieses erhöhte Risiko entspringt Code, der schwer zu verstehen, zu testen oder sicher zu ändern ist. Anders als direkte Fehlereinführung konzentriert sich dieses Problem auf die systematischen Faktoren, die Fehlereinführung wahrscheinlicher machen, was eine Umgebung schafft, in der selbst sorgfältige Entwickler wahrscheinlich Fehler machen.

## Indicators ⟡
- Fehlerraten steigen, wenn bestimmte Module oder Entwickler beteiligt sind
- Ähnliche Arten von Fehlern werden wiederholt in denselben Codebereichen eingeführt
- Code-Reviews erfassen häufig potenzielle Fehler, die Entwickler übersehen haben
- Entwickler äußern Unsicherheit über die Korrektheit ihrer Änderungen
- Testen offenbart Fehler, die während der Entwicklung offensichtlich hätten sein sollen

## Symptoms ▲

- [Hohe Rate an neu eingeführten Fehlern](hohe-rate-an-neu-eingefuehrten-fehlern.md)
<br/>  Wenn das Fehlerrisiko aufgrund von Codekomplexität erhöht ist, steigt die tatsächliche Rate, mit der Fehler eingeführt werden, messbar an.
- [Erhöhte Fehleranzahl](erhoehte-fehleranzahl.md)
<br/>  Ein höheres Fehlerrisiko führt direkt dazu, dass sich über die Zeit mehr Fehler im System anhäufen.
- [Angst vor Veränderung](angst-vor-veraenderung.md)
<br/>  Wenn Entwickler wissen, dass Änderungen wahrscheinlich Fehler einführen, werden sie zurückhaltend, die Codebasis zu ändern.
- [Erhöhte Entwicklungskosten](erhoehte-entwicklungskosten.md)
<br/>  Mehr Fehler bedeuten mehr Zeit für Debugging und Behebung, was die Entwicklungskosten in die Höhe treibt.
- [Ständiges Feuerlöschen](staendiges-feuerloeschen.md)
<br/>  Ein hohes Fehlerrisiko führt zu häufigen Produktionsproblemen, die dringende Aufmerksamkeit erfordern, was das Team im reaktiven Modus hält.

## Causes ▼

- [Schwer verständlicher Code](schwer-verstaendlicher-code.md)
<br/>  Code, der schwer zu verstehen ist, macht es viel wahrscheinlicher, dass Entwickler bei Änderungen Defekte einführen.
- [Komplexe und unklare Logik](komplexe-und-unklare-logik.md)
<br/>  Verworrene Geschäftslogik mit unklarer Absicht schafft Bedingungen, unter denen Fehler leicht eingeführt werden.
- [Unzureichendes Testen](unzureichendes-testen.md)
<br/>  Ohne automatisierte Tests, die Regressionen erfassen, trägt jede Codeänderung ein höheres Risiko, Fehler einzuführen.
- [Probleme durch enge Kopplung](probleme-durch-enge-kopplung.md)
<br/>  Eng gekoppelter Code bedeutet, dass Änderungen in einem Bereich andere Bereiche unvorhersehbar beeinflussen können, was das Risiko unbeabsichtigter Fehler erhöht.
- [Geringere Codequalität](geringere-codequalitaet.md)
<br/>  Minderwertiger Code mit inkonsistenten Mustern und schlechter Struktur erschwert es, über Korrektheit nachzudenken, was das Fehlerrisiko erhöht.
- [Unerfahrene Entwickler](unerfahrene-entwickler.md)
<br/>  Entwicklern ohne Erfahrung führen mit höherer Wahrscheinlichkeit Defekte durch Missverständnis von Code oder Geschäftslogik ein.
- [Anhäufung von Workarounds](anhaeufung-von-workarounds.md)
<br/>  Geschichtete Workarounds schaffen unerwartete Interaktionen und Randfälle zwischen alten und neuen Logikpfaden, was es wahrscheinlicher macht, dass Änderungen Fehler einführen.

## Detection Methods ○
- **Fehlermuster-Analyse:** Nachverfolgung, welche Codebereiche oder Arten von Änderungen am wahrscheinlichsten Fehler einführen
- **Entwicklerspezifische Metriken:** Überwachung der Fehlereinführungsraten einzelner Entwickler zur Identifikation von Schulungsbedarf
- **Korrelation von Codekomplexität:** Analyse der Beziehung zwischen Codekomplexitätsmetriken und Fehlerdichte
- **Änderungsauswirkungsanalyse:** Nachverfolgung, welche Arten von Änderungen am wahrscheinlichsten Probleme verursachen
- **Test-Wirksamkeit:** Messung, wie viele Fehler während der Entwicklung vs. in Produktion erfasst werden

## Examples

Ein Legacy-Bestandsverwaltungssystem hat ein Preisberechnungsmodul mit verschachtelter bedingter Logik, das Dutzende Sonderfälle für unterschiedliche Produkttypen, Kundenkategorien und Werbeaktionsrabatte handhabt. Die Logik ist über mehrere Funktionen mit unklarer Benennung und ohne Dokumentation verteilt, die die Geschäftsregeln erklärt. Wenn Entwickler Unterstützung für eine neue Produktkategorie hinzufügen müssen, müssen sie diese komplexe Logik navigieren, um zu verstehen, wo Änderungen vorzunehmen sind. Aufgrund der Komplexität übersehen sie häufig Randfälle oder missverstehen bestehende Regeln, was Fehler einführt, bei denen bestimmte Kombinationen von Produkten und Werbeaktionen falsche Preise produzieren. Trotz sorgfältiger Code-Reviews bleiben diese Fehler oft unentdeckt, weil auch Reviewer Schwierigkeiten haben, alle Interaktionen innerhalb der komplexen Preislogik zu verstehen. Ein weiteres Beispiel betrifft ein Nutzerauthentifizierungssystem, bei dem Passwortvalidierung, Session-Management und Berechtigungsprüfung in einer einzigen großen Klasse verflochten sind. Wenn Entwickler irgendein Authentifizierungsverhalten ändern müssen, müssen sie die gesamte Klasse und ihre vielen Verantwortlichkeiten verstehen. Die Komplexität macht es leicht, versehentlich unzusammenhängende Funktionalität zu brechen, etwa die Passwortvalidierungslogik zu ändern und unbeabsichtigt das Session-Timeout-Verhalten zu beeinflussen.
