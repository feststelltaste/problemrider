---
title: Fachliche Reviews
description: Arbeitsergebnisse von Fachexperten überprüfen und freigeben
  lassen.
category:
- Process
- Communication
problems:
- misaligned-deliverables
- requirements-ambiguity
- stakeholder-developer-communication-gap
- implementation-rework
- inadequate-requirements-gathering
- poor-domain-model
- inconsistent-behavior
- quality-blind-spots
layout: solution
lang: de
en_slug: subject-matter-reviews
related_solutions:
- slug: code-review-process-reform
  similarity: 0.75
- slug: domain-experts
  similarity: 0.7
- slug: architecture-reviews
  similarity: 0.7
- slug: on-site-customer
  similarity: 0.7
- slug: requirements-analysis
  similarity: 0.7
- slug: prototypes
  similarity: 0.7
---

## Description

Ein fachliches Review lässt Fachexperten ein abgeschlossenes Arbeitsergebnis gegen echte Geschäftsszenarien prüfen, statt gegen Abnahmekriterien, die Entwickler selbst geschrieben haben, sodass das Review Geschäftslogikfehler erfassen kann, die keine Menge technischen Testens zutage bringen würde. Dies ist besonders wichtig in der Legacy-Modernisierung, wo die wahre Spezifikation für das Verhalten eines Ersatzsystems häufig das stillschweigende Wissen der Personen ist, die das Legacy-System täglich nutzen, statt irgendetwas Dokumentiertes, was bedeutet, dass eine technisch korrekte Neuimplementierung dennoch still auf Weisen falsch sein kann, die nur ein Fachexperte bemerken würde. Experten durch den Ersatz mit ihren tatsächlichen täglichen Workflows zu führen und ihnen Nebeneinander-Vergleiche gegen Legacy-Ausgaben für dieselben Eingaben zu geben, bringt Diskrepanzen zutage, die automatisierte Regressionstests — gebaut aus denselben Annahmen wie der Code — nicht von sich aus erkennen können. Da Fachexperten typischerweise die beschäftigtsten Personen in der Organisation sind, macht die gute Planung dieser Reviews und ihre ausreichend straffe Strukturierung, um auf Qualität statt offene Feature-Diskussion fokussiert zu bleiben, die Praxis nachhaltig.

## How to Apply ◆

> In der Legacy-Modernisierung stellen fachliche Reviews sicher, dass Ersatzimplementierungen tatsächlich der Geschäftsabsicht hinter dem Legacy-Systemverhalten entsprechen, nicht nur seiner technischen Oberfläche.

- Planen Sie regelmäßige Review-Sitzungen, in denen Fachexperten abgeschlossene Features gegen echte Geschäftsszenarien prüfen, nicht nur gegen von Entwicklern geschriebene Abnahmekriterien.
- Lassen Sie Fachexperten durch das Ersatzsystem mit ihren tatsächlichen täglichen Workflows gehen, statt mit geskripteten Testfällen — dies offenbart Usability- und Korrektheitsprobleme, die formales Testing übersieht.
- Beziehen Sie fachliche Reviews an Schlüsselmeilensteinen der Modernisierung ein, besonders vor der Außerbetriebnahme jeder Legacy-Systemkomponente.
- Stellen Sie Fachexperten Nebeneinander-Vergleiche zwischen Legacy- und Ersatzsystem-Ausgaben für dieselben Eingaben bereit, was Diskrepanzen sofort sichtbar macht.
- Dokumentieren Sie Feedback von Fachexperten systematisch und verfolgen Sie die Lösung, um Vertrauen aufzubauen, dass Bedenken adressiert statt ignoriert werden.
- Wählen Sie Reviewer, die verschiedene Nutzergruppen und Erfahrungsstufen repräsentieren, um diverse Perspektiven auf das Ersatzsystem zu erfassen.

## Tradeoffs ⇄

> Fachliche Reviews erfassen geschäftskritische Probleme, die technische Reviews übersehen, erfordern aber Zugang zu beschäftigten Fachexperten und sorgfältige Planung.

**Vorteile:**

- Erfasst Geschäftslogikfehler, die automatisierte Tests und Code-Reviews übersehen, weil sie Fachexpertise zur Identifikation erfordern.
- Baut Vertrauen der Fachexperten in die Modernisierungsanstrengung auf, indem ihnen eine Stimme in der Qualitätssicherung gegeben wird.
- Bringt undokumentierte Geschäftsregeln und Grenzfälle zutage, die Fachexperten intuitiv kennen, aber nie niedergeschrieben haben.
- Reduziert das Risiko, einen technisch korrekten, aber geschäftlich fehlerhaften Ersatz auszuliefern.

**Kosten und Risiken:**

- Fachexperten sind oft die beschäftigtsten Personen in der Organisation, was die Planung regelmäßiger Review-Sitzungen erschwert.
- Reviews ohne klare Struktur können zu Umfangsdiskussionen oder Feature-Anfragen statt Qualitätsvalidierung entarten.
- Fachexperten verstehen möglicherweise nicht die technischen Einschränkungen, die Designentscheidungen beeinflussten, was zu Feedback führt, das schwer umzusetzen ist.
- Übermäßiges Vertrauen auf eine kleine Anzahl von Fachexperten schafft einen Wissensengpass und Single Point of Failure im Review-Prozess.

## How It Could Be

> Das folgende Szenario veranschaulicht den Wert fachlicher Reviews während der Legacy-Modernisierung.

Ein Logistikunternehmen ersetzte seine Fracht-Bewertungs-Engine, und automatisierte Tests zeigten 99,8 % Übereinstimmung mit den Berechnungen des Legacy-Systems. Während eines fachlichen Reviews bemerkte jedoch eine leitende Tarifanalystin, dass das Ersatzsystem Treibstoffzuschläge vor statt nach volumetrischen Anpassungen anwendete — ein subtiles Reihenfolgeproblem, das die Testdaten nicht aufgedeckt hatten, weil die meisten Testsendungen unter dem volumetrischen Schwellenwert lagen. Die Analystin schätzte, dass dieser Fehler ungefähr 2 Millionen Dollar an jährlichem Umsatzverlust über hochvolumige Versandrouten hinweg verursacht hätte. Dieser einzelne Befund rechtfertigte die gesamte Investition in fachliche Reviews für das Projekt.
