---
title: Schwer verständlicher Code
description: Es ist schwer, den Zweck von Modulen oder Funktionen zu erfassen, ohne
  viele andere Teile des Systems zu verstehen, was die Entwicklung verlangsamt und
  Fehler begünstigt.
category:
- Code
related_problems:
- slug: difficult-code-comprehension
  similarity: 0.85
- slug: complex-and-obscure-logic
  similarity: 0.75
- slug: difficult-to-test-code
  similarity: 0.75
- slug: increased-cognitive-load
  similarity: 0.7
- slug: debugging-difficulties
  similarity: 0.7
- slug: difficult-code-reuse
  similarity: 0.7
solutions:
- clean-code
- design-by-contract
- loose-coupling
- code-comments
- fluent-interfaces
- code-reading-sessions
- preparatory-refactoring
- code-conventions
- communities-of-practice
- internal-technical-coaching
- ubiquitous-language
- technical-debt-assessment
- duplication-detection
- attribute-usage-analysis
- typed-schema-extraction
layout: problem
lang: de
en_slug: difficult-to-understand-code
---

## Description

Schwer verständlicher Code entsteht, wenn Softwarekomponenten so implementiert sind, dass ihr Zweck, Verhalten oder Zusammenspiel für Entwickler unklar bleibt, die mit ihnen arbeiten müssen. Dieses Problem äußert sich als Code, der umfangreichen Kontext erfordert, unklare Benennung hat, nicht offensichtlichen Logikmustern folgt oder keine ausreichende Dokumentation hat, um seine beabsichtigte Funktion zu verstehen. Schwer verständlicher Code verlangsamt die Entwicklung erheblich und erhöht die Wahrscheinlichkeit, Fehler einzuführen.

## Indicators ⟡

- Entwickler verbringen übermäßig viel Zeit damit, zu verstehen, was Code tut, bevor sie ihn ändern
- Code-Reviews erfordern langwierige Erklärungen der Implementierungslogik
- Neue Teammitglieder haben Schwierigkeiten, bestehende Codefunktionalität zu verstehen
- Dokumentation oder Kommentare sind nötig, um grundlegende Codeoperationen zu erklären
- Ähnliche Funktionalität ist über die Codebasis hinweg unterschiedlich implementiert

## Symptoms ▲

- [Schwieriges Onboarding neuer Entwickler](schwieriges-onboarding-neuer-entwickler.md)
<br/>  Code, der schwer zu verstehen ist, lässt neue Entwickler viel länger brauchen, um das System zu erlernen und produktiv zu werden.
- [Erhöhtes Risiko für Fehler](erhoehtes-risiko-fuer-fehler.md)
<br/>  Entwickler, die Code nicht vollständig verstehen, treffen mit höherer Wahrscheinlichkeit falsche Annahmen und führen Fehler ein.
- [Langsame Entwicklungsgeschwindigkeit](langsame-entwicklungsgeschwindigkeit.md)
<br/>  Entwickler verbringen übermäßig viel Zeit damit, Code zu lesen und zu verstehen, bevor sie Änderungen vornehmen können, was die Geschwindigkeit verlangsamt.
- [Anhäufung von Workarounds](anhaeufung-von-workarounds.md)
<br/>  Wenn Entwickler bestehenden Code nicht gut genug verstehen, um ihn korrekt zu ändern, fügen sie stattdessen Workarounds hinzu.
- [Große Schätzungen für kleine Änderungen](grosse-schaetzungen-fuer-kleine-aenderungen.md)
<br/>  Einfache Änderungen erfordern große Schätzungen, weil Entwickler zunächst erhebliche Zeit damit verbringen müssen, den umgebenden Code zu verstehen.

## Causes ▼

- [Schlechte Namenskonventionen](schlechte-namenskonventionen.md)
<br/>  Unklare Variablen- und Funktionsnamen verschleiern Zweck und Verhalten des Codes.
- [Übertrieben cleverer Code](uebertrieben-cleverer-code.md)
<br/>  Code, der geschrieben wurde, um clever statt klar zu sein, opfert Lesbarkeit für Kürze oder Eleganz.
- [Komplexe und unklare Logik](komplexe-und-unklare-logik.md)
<br/>  Geschäftslogik, die durch verworrene Muster umgesetzt wurde, macht es extrem schwer, der Absicht des Codes zu folgen.
- [Inkonsistente Coding-Standards](inkonsistente-coding-standards.md)
<br/>  Inkonsistente Codemuster über die Codebasis hinweg hindern Entwickler daran, zuverlässige mentale Modelle zu bilden.
- [Informationsverfall](informationsverfall.md)
<br/>  Wenn Dokumentation veraltet ist oder fehlt, haben Entwickler keine Referenz, um die ursprüngliche Design-Absicht zu verstehen.

## Detection Methods ○

- **Code-Review-Feedback-Analyse:** Beobachtung, wie oft Reviewer um Klärung der Codefunktionalität bitten
- **Entwickler-Zeittracking:** Messung der Zeit, die für das Verstehen vs. das Ändern bestehenden Codes aufgewendet wird
- **Code-Komplexitätsmetriken:** Nutzung statischer Analysewerkzeuge zur Identifikation übermäßig komplexen oder schwer verständlichen Codes
- **Onboarding-Feedback:** Befragung neuer Teammitglieder zu Herausforderungen beim Codeverständnis
- **Dokumentationslücken-Analyse:** Identifikation von Codebereichen, denen ausreichende Erklärung fehlt

## Examples

Ein Datenverarbeitungsmodul nutzt Variablennamen wie `proc1`, `proc2` und `tempData` ohne Kommentare, die beschreiben, welche Art von Verarbeitung stattfindet oder was die temporären Daten repräsentieren. Das Verständnis, wie das Modul geändert werden kann, erfordert das Nachverfolgen mehrerer Funktionen und das Lesen von Datenbankabfragen, um die tatsächliche umgesetzte Geschäftslogik abzuleiten. Ein weiteres Beispiel betrifft ein Authentifizierungssystem, bei dem der Login-Ablauf durch sechs unterschiedliche Klassen mit Namen wie `AuthManager`, `AuthHandler`, `AuthProcessor` und `AuthController` läuft, von denen jede ähnlich klingende, aber unterschiedliche Funktionen ausführt, was es extrem schwierig macht, den gesamten Authentifizierungsprozess zu verstehen oder zu identifizieren, wo bestimmte Funktionalität implementiert ist.
