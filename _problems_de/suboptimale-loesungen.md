---
title: Suboptimale Lösungen
description: Gelieferte Lösungen funktionieren, sind aber ineffizient, schwer zu nutzen
  oder adressieren die zugrunde liegenden Probleme, die sie lösen sollten, nicht
  vollständig.
category:
- Architecture
- Code
- Requirements
related_problems:
- slug: accumulation-of-workarounds
  similarity: 0.6
- slug: process-design-flaws
  similarity: 0.6
- slug: workaround-culture
  similarity: 0.55
- slug: increased-technical-shortcuts
  similarity: 0.55
- slug: complex-implementation-paths
  similarity: 0.55
- slug: second-system-effect
  similarity: 0.55
solutions:
- architecture-reviews
- boring-technologies
- clean-code
- design-by-contract
- pattern-language
- domain-patterns
- domain-immersion
- lightweight-design-review
layout: problem
lang: de
en_slug: suboptimal-solutions
---

## Description

Suboptimale Lösungen treten auf, wenn implementierte Systeme oder Prozesse technisch funktionieren, aber hinter dem zurückbleiben, was mit besserem Design, Anforderungsanalyse oder Implementierungsansätzen erreicht werden könnte. Diese Lösungen lösen möglicherweise unmittelbare Probleme, schaffen aber Ineffizienzen, Nutzerfrustration oder Wartungslasten, die ein durchdachterer Ansatz hätte vermeiden können.

## Indicators ⟡

- Lösungen funktionieren, erfordern aber exzessive Schritte oder Aufwand von Nutzern
- Workarounds sind nötig, um übliche Aufgaben zu erledigen
- Die Performance ist angemessen, aber viel langsamer als nötig
- Lösungen adressieren Symptome statt Grundursachen
- Nutzer äußern, dass „es einen besseren Weg geben muss", um Aufgaben zu erledigen

## Symptoms ▲

- [Anhäufung von Workarounds](anhaeufung-von-workarounds.md)
<br/>  Nutzer und Entwickler erstellen Workarounds, um die Ineffizienzen und Lücken in suboptimalen Lösungen zu kompensieren.
- [Nutzerfrustration](nutzerfrustration.md)
<br/>  Nutzer werden frustriert, wenn Lösungen umständlich, ineffizient sind oder ihre Bedürfnisse nicht vollständig erfüllen.
- [Unzufriedenheit der Stakeholder](unzufriedenheit-der-stakeholder.md)
<br/>  Stakeholder sind enttäuscht, wenn gelieferte Lösungen die zugrunde liegenden Geschäftsbedürfnisse, die sie lösen sollten, nicht vollständig erfüllen.
- [Hohe Wartungskosten](hohe-wartungskosten.md)
<br/>  Suboptimale Designs erfordern anhaltende Workarounds, Patches und Support, die Wartungskosten aufblähen.
- [Langsame Anwendungsperformance](langsame-anwendungsperformance.md)
<br/>  Ineffiziente Lösungsdesigns äußern sich als schlechte Performance, die Nutzer beobachten und messen können.

## Causes ▼

- [Unzureichende Anforderungserhebung](unzureichende-anforderungserhebung.md)
<br/>  Unzureichendes Verständnis tatsächlicher Nutzerbedürfnisse führt zu Lösungen, die um falsche Annahmen herum designt sind.
- [Termindruck](termindruck.md)
<br/>  Zeitdruck zwingt Teams, die erste funktionierende Lösung statt der besten Lösung zu liefern.
- [Wissenslücken](wissensluecken.md)
<br/>  Mangel an Domänen- oder technischem Wissen führt dazu, dass Entwickler Ansätze wählen, die funktionieren, aber weit von optimal entfernt sind.
- [Annahmenbasierte Entwicklung](annahmenbasierte-entwicklung.md)
<br/>  Der Bau von Lösungen basierend auf unvalidierten Annahmen über Nutzerbedürfnisse produziert Features, die das Ziel verfehlen.
- [Angehäufte Entscheidungsschulden](angehaeufte-entscheidungsschulden.md)
<br/>  Wenn ein Rückstand aufgeschobener, voneinander abhängiger Entscheidungen schließlich unter Druck gelöst werden muss, zwingen die verbleibenden eingeschränkten Optionen oft zu suboptimalen Entscheidungen.

## Detection Methods ○

- **Bewertung der Nutzererfahrung:** Bewertung, wie effizient Nutzer Aufgaben mit gelieferten Lösungen erledigen können
- **Performance-Benchmarking:** Vergleich der Lösungsperformance mit Industriestandards oder Alternativen
- **Usability-Tests:** Testen von Lösungen mit echten Nutzern zur Identifikation von Ineffizienzen
- **Kosten-Nutzen-Analyse:** Bewertung, ob Lösungen den erwarteten Wert im Vergleich zu Alternativen bieten
- **Skalierbarkeitstests:** Bewertung, ob Lösungen erwartetes Wachstum handhaben können

## Examples

Ein Dokumentenmanagementsystem erfordert, dass Nutzer 12 Klicks ausführen und durch 4 verschiedene Bildschirme navigieren, um eine Aufgabe zu erledigen, die 2 Klicks dauern sollte, weil das System um die Datenbankstruktur herum statt um den Nutzer-Workflow designt wurde. Während das System Nutzern technisch erlaubt, Dokumente zu verwalten, ist es so umständlich, dass die Produktivität tatsächlich im Vergleich zum vorherigen papierbasierten Prozess sinkt. Ein weiteres Beispiel betrifft eine Datenintegrationslösung, die manuelles Eingreifen erfordert, jedes Mal wenn neue Datenquellen hinzugefügt werden, obwohl die Anforderung klar besagte, dass das System neue Datenquellen automatisch handhaben sollte — die Lösung funktioniert, schafft aber anhaltende operative Last.
