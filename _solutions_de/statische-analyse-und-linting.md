---
title: Statische Analyse und Linting
description: Automatische Überprüfung von Quellcode auf potenzielle
  Probleme.
category:
- Code
- Process
quality_tactics_url: https://qualitytactics.de/en/maintainability/static-code-analysis/
problems:
- inconsistent-coding-standards
- inconsistent-naming-conventions
- poor-naming-conventions
- mixed-coding-styles
- undefined-code-style-guidelines
- inconsistent-codebase
- hardcoded-values
- null-pointer-dereferences
- integer-overflow-underflow
- unreleased-resources
- style-arguments-in-code-reviews
- automated-tooling-ineffectiveness
- database-connection-leaks
- improper-event-listener-management
- inadequate-initial-reviews
- nitpicking-culture
- perfectionist-review-culture
- stack-overflow-errors
- bikeshedding
- circular-references
- conflicting-reviewer-opinions
- copy-paste-programming
- race-conditions
- deadlock-conditions
- log-injection-vulnerabilities
layout: solution
lang: de
en_slug: static-analysis-and-linting
related_solutions:
- slug: code-metrics
  similarity: 0.85
- slug: code-review-process-reform
  similarity: 0.85
- slug: static-code-analysis
  similarity: 0.8
- slug: code-quality-gates
  similarity: 0.8
- slug: dynamic-code-analysis
  similarity: 0.8
- slug: code-coverage-analysis
  similarity: 0.8
---

## Description

Statische Codeanalyse durchsucht Quellcode automatisch nach bekannten Problemmustern — Null-Dereferenzierungen, hartcodierte Zugangsdaten, übermäßige Komplexität —, ohne dass ein menschlicher Reviewer zunächst verstehen muss, was der Code tut, was genau das ist, was sie gegen eine Legacy-Codebasis wertvoll macht, die niemand mehr vollständig versteht. Ein erster Scan eines echt alten Systems bringt routinemäßig Tausende von Befunden zutage, weshalb der richtige erste Schritt darin besteht, diese Anzahl einzufrieren, statt sie als unmittelbare To-Do-Liste zu behandeln: neuen Code gegen die Einführung weiterer Verstöße absichern, dann den Rückstand opportunistisch als Schulden-Inventar abarbeiten statt als Notfall. Der blinde Fleck des Werkzeugs ist jedoch real — es kann ein Ressourcenleck oder eine Namensinkonsistenz markieren, aber es hat keine Möglichkeit, eine falsche Geschäftsannahme oder eine unpassende Abstraktion zu erkennen, was genau die Art von Design-Ebenen-Schulden ist, die in älteren Systemen tendenziell dominiert.

## How to Apply ◆

> In einer Legacy-Codebasis ist statische Analyse eines der wenigen Werkzeuge, die den vollen Umfang jahrelang angesammelter Probleme scannen können, ohne dass jemand den Code zuerst verstehen muss.

- Beginnen Sie mit einem "Nur neuer Code"-Modus, der Regeln auf geänderte Zeilen durchsetzt, während bestehende Verstöße unterdrückt werden — dies verhindert, dass sich die Codebasis verschlechtert, ohne das Team mit Tausenden bereits bestehender Befunde zu überwältigen.
- Führen Sie einen vollständigen Baseline-Scan auf der Legacy-Codebasis durch, um ein Inventar von Verstößen zu generieren, und behandeln Sie dieses Inventar dann als Schulden-Rückstand statt als To-Do-Liste, die sofortiges Handeln erfordert.
- Priorisieren Sie sicherheitsfokussierte Regeln zuerst (SQL-Injection-Muster, hartcodierte Zugangsdaten, unescapte Eingaben), weil Legacy-Systeme oft modernem Sicherheitsbewusstsein vorausgehen und unentdeckte Schwachstellen tragen.
- Integrieren Sie Analyse in die CI-Pipeline, sodass jeder Pull Request daran gemessen wird, keine neuen Verstöße einzuführen — auch wenn bestehende Verstöße bleiben, darf die Anzahl nicht wachsen.
- Nutzen Sie Komplexitätsmetriken (zyklomatische Komplexität, kognitive Komplexität), um die Module zu identifizieren, in denen das Änderungsrisiko am höchsten ist; diese Hotspots sind, wo Legacy-Vorfälle am häufigsten entstehen.
- Aktivieren Sie Dead-Code-Erkennung, um unreferenzierte Funktionen und Klassen zu finden, die sich ansammelten, während Features über Jahrzehnte ersetzt wurden — das Entfernen von totem Code reduziert die Fläche, die Entwickler verstehen müssen.
- Setzen Sie architektonische Grenzregeln mit Werkzeugen wie ArchUnit oder Dependency Cruiser durch, um weitere Verstöße gegen die Modulstruktur zu erkennen und zu verhindern, die die ursprünglichen Architekten beabsichtigten.
- Planen Sie periodische Scans der gesamten Codebasis (nächtlich oder wöchentlich), um Verstöße in Code zu erfassen, der vor den aktuellen Standards liegt, und um zu verfolgen, ob das gesamte Schuldenniveau steigt oder sinkt.

## Tradeoffs ⇄

> Statische Analyse bietet objektive, skalierbare Qualitätsmessung für Legacy-Systeme, aber ihr Wert hängt vollständig von der Bereitschaft des Teams ab, auf Befunde zu reagieren, statt sie zu unterdrücken oder zu ignorieren.

**Vorteile:**

- Bringt Qualitätsprobleme über die gesamte Legacy-Codebasis systematisch zutage, einschließlich in Modulen, die kein aktuelles Teammitglied gelesen oder angefasst hat.
- Bietet objektive, quantifizierte Metriken (Technische-Schulden-Verhältnis, Verstoßanzahlen, Komplexitätswerte), die es möglich machen, gegenüber nicht-technischen Stakeholdern für Sanierungsinvestition zu argumentieren.
- Erfasst wiederkehrende Fehlermuster — Null-Dereferenzierungen, Ressourcenlecks, veraltete API-Nutzung —, die häufig in Legacy-Code auftreten, der vor der Etablierung aktueller Praktiken geschrieben wurde.
- Befreit menschliche Reviewer von mechanischen Prüfungen, sodass sie sich auf Design- und Geschäftslogikprobleme konzentrieren können, die automatisierte Werkzeuge nicht bewerten können.
- Schafft eine Qualitätsbaseline, die es möglich macht zu messen, ob Modernisierungsanstrengungen die Codebasis tatsächlich verbessern oder nur Probleme umordnen.

**Kosten und Risiken:**

- Legacy-Codebasen erzeugen typischerweise Tausende von Verstößen beim ersten Scan; ohne Triage-Strategie ist das Volumen lähmend, und Teams könnten das Werkzeug deaktivieren, statt Befunde zu adressieren.
- Falsch positive Ergebnisse in Legacy-Code sind häufiger, weil der Code oft Muster nutzt, die den von den Werkzeugen erwarteten Konventionen vorausgehen, was erhebliche Regelanpassung erfordert, um Rauschen zu reduzieren.
- Analysewerkzeuge fügen CI-Pipelines Zeit hinzu; in Legacy-Systemen mit langsamen Build-Prozessen könnte das Hinzufügen eines weiteren Schritts zu einer bereits langsamen Pipeline Entwicklerwiderstand erzeugen.
- Teams, die auf Analysewerte statt echte Qualität optimieren, werden Befunde unterdrücken oder Code umstrukturieren, um die Metrik zu erfüllen, ohne das zugrunde liegende Verhalten zu verbessern.
- Statische Analyse kann keine Geschäftslogikfehler, falsche Domänenannahmen oder die Art von Design-Ebenen-Schulden (falsche Abstraktionen, unpassende Kopplung) erkennen, die viele Legacy-Systeme dominiert.

## How It Could Be

> Die folgenden Szenarien veranschaulichen, wie statische Analyse in echten Legacy-Modernisierungsanstrengungen eingeführt und genutzt wird.

Ein Logistikunternehmen erbte eine zehn Jahre alte PHP-Anwendung, die von aufeinanderfolgenden Entwicklungsagenturen ohne konsistente Coding-Standards erweitert worden war. Als das neue interne Team zum ersten Mal einen statischen Analyse-Scan durchführte, produzierte er über 8.000 Befunde. Statt zu versuchen, alle zu beheben, kategorisierte das Team Befunde nach Schweregrad und fror die bestehende Anzahl ein — jeder neue Code musste die Verstoßanzahl unverändert lassen oder senken. Nach sechs Monaten stetiger inkrementeller Bereinigung während regulärer Feature-Arbeit war die Anzahl auf unter 3.000 gesunken, und das Team hatte jeden kritischen Sicherheitsbefund beseitigt. Die Metriken gaben dem Management eine konkrete Möglichkeit, Fortschritt zu sehen, ohne Code lesen zu müssen.

Das Schadensbearbeitungssystem eines Versicherungsunternehmens war in Java geschrieben und hatte seit Jahren keine architektonische Aufmerksamkeit erhalten. Das Team fügte ArchUnit-Tests hinzu, die die beabsichtigten Schichtungsregeln kodierten — Services dürfen nicht direkt auf die Datenbankschicht zugreifen, Domänenobjekte dürfen nicht auf Infrastrukturklassen verweisen — und führte sie in CI aus. Beim ersten Lauf erschienen 47 Verstöße, die meisten davon in Modulen, die organisch über ihre ursprünglichen Grenzen hinausgewachsen waren. Das Team nutzte diese Liste, um seinen Refactoring-Rückstand zu priorisieren, wobei die Verstöße in der Reihenfolge der während der Feature-Entwicklung am häufigsten geänderten Module abgearbeitet wurden.

Ein Telekommunikationsanbieter wollte eine C++-Abrechnungs-Engine modernisieren, hatte aber keine Ahnung, welche Teile der Codebasis noch aktiv ausgeführt wurden. Dead-Code-Erkennung offenbarte, dass ungefähr 30 % der Funktionen im Abrechnungsmodul von keinem erreichbaren Einstiegspunkt jemals aufgerufen wurden — sie waren Überbleibsel von Abrechnungsmodellen, die Jahre zuvor stillgelegt worden waren. Das Entfernen dieses Codes reduzierte die zu pflegende Fläche erheblich und machte die verbleibende Logik leichter zu verstehen und zu testen, was wiederum die geschätzten Kosten der geplanten Neuschreibung reduzierte.
