---
title: Funktionaler Spike
description: Untersuchung geschäftlicher Risiken durch zeitlich begrenzte Experimente.
category:
- Process
- Requirements
problems:
- fear-of-change
- analysis-paralysis
- modernization-strategy-paralysis
- difficulty-quantifying-benefits
- assumption-based-development
- implementation-rework
- fear-of-breaking-changes
- history-of-failed-changes
- inability-to-innovate
layout: solution
lang: de
en_slug: functional-spike
related_solutions:
- slug: technical-spike
  similarity: 0.8
- slug: prototyping
  similarity: 0.75
- slug: prototypes
  similarity: 0.75
- slug: walking-skeleton
  similarity: 0.7
- slug: functional-tests
  similarity: 0.7
- slug: requirements-analysis
  similarity: 0.7
---

## Description

Ein funktionaler Spike ist ein kurzes, strikt zeitlich begrenztes Experiment, das darauf ausgelegt ist, eine spezifische, risikoreiche Frage zu beantworten, bevor sich ein Team zu einem größeren Arbeitsumfang verpflichtet — zum Beispiel, ob eine Geschäftsregel aus einem Monolithen extrahiert werden kann, ohne abhängige Workflows zu brechen, oder ob eine moderne Engine Jahre handjustierten Legacy-Verhaltens replizieren kann. Anders als ein Proof of Concept, der auf Produktionsqualität abzielt, produziert ein Spike bewusst Wegwerfcode; sein Ergebnis ist nicht Software, sondern Evidenz, in Form einer konkreten Antwort, die Annahme durch Beobachtung ersetzt. Dies unterscheidet ihn von einem technischen Spike, der Umsetzungsmachbarkeit untersucht, indem er sich stattdessen auf geschäftliches Risiko fokussiert: ob eine vorgeschlagene Änderung das Verhalten bewahrt, auf das sich Stakeholder tatsächlich verlassen. Legacy-Systeme neigen besonders dazu, undokumentierte Geschäftslogik und versteckte Abhängigkeiten anzusammeln, die erst auftauchen, wenn jemand tatsächlich eine Änderung versucht — genau die Art Risiko, die ein Spike dazu gebaut ist, günstig und früh sichtbar zu machen. Indem sie eine kostspielige Verpflichtung in ein kleines, begrenztes Experiment umwandeln, lassen funktionale Spikes Teams Unmachbarkeit, versteckte Kopplung oder unerwartete Komplexität innerhalb von Tagen statt nach Monaten einer gescheiterten Migration entdecken, und sie geben Stakeholdern konkrete Evidenz, die sie gegen die Alternative anhaltender Analyse-Paralyse abwägen können.

## How to Apply ◆

> In Legacy-Systemen helfen funktionale Spikes Teams, Unsicherheit zu verringern, bevor sie sich zu kostspieligen Änderungen verpflichten, indem sie kurze, fokussierte Experimente durchführen.

- Identifizieren Sie die risikoreichsten Annahmen in einem geplanten Legacy-Modernisierungsvorhaben — zum Beispiel, ob eine kritische Geschäftsregel aus einem Monolithen extrahiert werden kann, ohne abhängige Workflows zu brechen.
- Setzen Sie jedem Spike eine strikte Zeitbox (ein bis fünf Tage) und definieren Sie eine klare Frage, die er beantworten muss, wie „Können wir die Batch-Preisberechnung durch einen Echtzeitdienst ersetzen, ohne Latenzschwellen zu überschreiten?"
- Bauen Sie die einfachste mögliche Implementierung, die die Frage beantwortet — Wegwerfcode ist akzeptabel und erwartet, weil das Ziel Lernen ist, nicht Produktionsreife.
- Beziehen Sie Fachexperten während des Spikes ein, um zu validieren, dass das Experiment echtes Geschäftsverhalten adressiert, besonders wenn Legacy-Logik undokumentiert ist oder nur in Erfahrungswissen existiert.
- Dokumentieren Sie die Befunde sofort nach Abschluss des Spikes, einschließlich was funktioniert hat, was fehlgeschlagen ist und welche neuen Risiken entdeckt wurden.
- Nutzen Sie Spike-Ergebnisse, um Schätzungen und Pläne für die tatsächliche Umsetzung zu aktualisieren und Vermutungen durch Evidenz aus dem Experiment zu ersetzen.
- Wenn ein Spike offenbart, dass der ursprüngliche Ansatz nicht tragfähig ist, behandeln Sie das als Erfolg — das Team hat Wochen oder Monate verschwendeten Aufwands vermieden.

## Tradeoffs ⇄

> Spikes tauschen eine kleine Zeitmenge gegen deutlich verringertes Risiko, erfordern aber Disziplin, um fokussiert und zeitlich begrenzt zu bleiben.

**Vorteile:**

- Verringert das Risiko, sich zu teuren Modernisierungspfaden zu verpflichten, die sich als unmachbar erweisen, indem technische und geschäftliche Blocker früh sichtbar gemacht werden.
- Baut Teamvertrauen in vorgeschlagene Änderungen auf, indem konkrete Evidenz statt theoretischer Argumente bereitgestellt wird.
- Hilft, Modernisierungsinvestitionen gegenüber Stakeholdern zu rechtfertigen, indem Machbarkeit mit minimalen Vorabkosten demonstriert wird.
- Deckt versteckte Abhängigkeiten und undokumentiertes Verhalten in Legacy-Systemen auf, bevor sie vollständige Umsetzungsvorhaben entgleisen lassen.

**Kosten und Risiken:**

- Teams könnten sich schwertun, Spike-Code zu verwerfen, und darauf bestehen, ihn zu Produktionscode weiterzuentwickeln, was die Qualität untergräbt.
- Ohne strikte Zeitbegrenzung können Spikes zu Mini-Projekten anwachsen, die Ressourcen verbrauchen, ohne Produktionswert zu liefern.
- Mit der Praxis nicht vertraute Stakeholder könnten Spikes als verschwendete Zeit statt als Risikominderung ansehen.
- Wiederholte Spikes ohne Weiterverfolgung in tatsächliche Umsetzung können Teammotivation und Stakeholder-Vertrauen untergraben.

## How It Could Be

> Die folgenden Szenarien veranschaulichen, wie funktionale Spikes Risiko in Legacy-Modernisierungskontexten verringern.

Ein Finanzdienstleistungsunternehmen musste bestimmen, ob sein 15 Jahre altes Auftragsverwaltungssystem in Microservices zerlegt werden konnte. Statt sich zu einem sechsmonatigen Migrationsplan zu verpflichten, führte das Team einen dreitägigen Spike durch, um eine einzelne Auftragsvalidierungsregel aus dem Monolithen zu extrahieren und als eigenständigen Dienst zu deployen. Der Spike offenbarte, dass die Validierungslogik von sieben undokumentierten Datenbank-Views und zwei gespeicherten Prozeduren abhing, was die Extraktion weit komplexer machte als anfänglich geschätzt. Dieser Befund veranlasste das Team, stattdessen einen Strangler-Fig-Ansatz zu übernehmen, was Monate an Nacharbeit sparte.

Eine E-Commerce-Plattform erwog, ihre Legacy-Suchmaschine durch eine moderne Alternative zu ersetzen. Ein zweitägiger Spike, der die Qualität der Suchergebnisse zwischen der alten und neuen Engine anhand von Produktionsdaten verglich, offenbarte, dass das Legacy-System über Jahre handjustierte Relevanz-Boosting-Regeln angesammelt hatte, die die neue Engine nicht von Haus aus replizieren konnte. Das Team nutzte diesen Befund, um eine gestaffelte Migration mit expliziten Relevanz-Tuning-Meilensteinen zu planen statt eines Big-Bang-Ersatzes.
