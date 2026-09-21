---
title: Regressionstests
description: Erneutes Ausführen bestehender Tests nach jeder Änderung gegen
  unbeabsichtigte Brüche.
category:
- Testing
- Code
quality_tactics_url: https://qualitytactics.de/en/functional-suitability/regression-testing/
problems:
- increased-bug-count
- increased-risk-of-bugs
- delayed-bug-fixes
- maintenance-paralysis
- large-estimates-for-small-changes
- reduced-code-submission-frequency
- rapid-system-changes
- increased-cost-of-development
- slow-development-velocity
- customer-dissatisfaction
- user-trust-erosion
- increased-manual-testing-effort
- outdated-tests
- upgrade-blocked-by-customization
layout: solution
lang: de
en_slug: regression-testing
related_solutions:
- slug: test-coverage-strategy
  similarity: 0.85
- slug: automated-tests
  similarity: 0.8
- slug: functional-tests
  similarity: 0.75
- slug: characterization-tests
  similarity: 0.75
- slug: code-quality-gates
  similarity: 0.75
- slug: ci-cd-pipeline
  similarity: 0.75
---

## Description

Regressionstests führen eine bestehende Testsuite nach jeder Änderung erneut aus, um unbeabsichtigte Brüche zu erfassen, und verwandeln das sonst unsichtbare Risiko einer Legacy-Modifikation in etwas Sichtbares, bevor es Produktion erreicht. Ohne dieses Sicherheitsnetz operiert ein Legacy-Team in einem Zustand permanenter Unsicherheit — jede Änderung könnte still etwas kaputt machen, und der einzige Weg, es herauszufinden, ist, wenn ein Nutzer es meldet —, was genau das ist, was Wartungslähmung produziert, bei der das Team Angst bekommt, Code anzufassen, der sich wirklich ändern muss. Der Aufbau der Suite erfordert nicht vollständige Abdeckung vom ersten Tag an: Mit Characterization Tests auf den spezifischen Bereichen zu beginnen, die das Team gerade modifiziert, und einen Regressionstest für jede Fehlerbehebung zu schreiben, sodass er nie still zurückkehren kann, lässt das Sicherheitsnetz genau dort wachsen, wo es am meisten gebraucht wird.

## How to Apply ◆

> Legacy-Systeme ohne Regressionstests operieren in einem Zustand permanenter Unsicherheit: Jede Änderung könnte etwas kaputt machen, aber das Team hat keine Möglichkeit, es zu wissen, bis Nutzer Probleme melden. Regressionstests in eine Legacy-Codebasis einzuführen bedeutet nicht, vom ersten Tag an 100 % Abdeckung zu erreichen — es geht darum, systematisch ein Sicherheitsnetz aufzubauen, das selbstbewusste Modifikation der kritischsten und am häufigsten geänderten Bereiche ermöglicht.

- Beginnen Sie damit, die kritischsten nutzerseitigen Arbeitsabläufe des Systems zu identifizieren und Ende-zu-Ende-Regressionstests zu schreiben, die verifizieren, dass diese Arbeitsabläufe korrekte Ergebnisse produzieren. In Legacy-Systemen bietet selbst eine kleine Suite von Tests, die die zehn wichtigsten Nutzer-Arbeitsabläufe abdeckt, mehr Wert als umfangreiche Unit-Tests in selten modifiziertem Code, weil sie direkt die Erosion des Nutzervertrauens adressiert, die durch Änderungen verursacht wird, die sichtbare Funktionalität brechen.
- Schreiben Sie Characterization Tests für die Codebereiche, die das Team modifizieren muss, aber nicht anzufassen wagt. Characterization Tests erfassen das aktuelle Verhalten des Systems — einschließlich Fehler und undokumentierter Seiteneffekte — als Basislinie. Sie behaupten nicht, dass das Verhalten korrekt ist; sie behaupten, dass es sich nicht geändert hat, was genau das Sicherheitsnetz ist, das gebraucht wird, um Wartungslähmung zu überwinden.
- Integrieren Sie Regressionstests in die CI-Pipeline, sodass sie automatisch bei jedem Pull Request laufen. Diese Integration ist der Mechanismus, der Tests von einem manuellen Verifikationsschritt (der unter Druck übersprungen wird) in ein verpflichtendes Qualitäts-Gate verwandelt. Für Legacy-Systeme, bei denen die vollständige Testsuite langsam ist, führen Sie eine schnelle Untermenge kritischer Pfade bei jedem PR aus und die vollständige Suite nächtlich.
- Schreiben Sie beim Beheben eines Fehlers einen Regressionstest, der den Fehler reproduziert, bevor Sie den Fix implementieren. Diese Praxis stellt sicher, dass behobene Fehler behoben bleiben, und adressiert direkt das Problem verzögerter Fehlerbehebungen, bei dem dieselben Probleme wieder auftauchen, nachdem sie gelöst wurden. Über die Zeit baut dieser Ansatz eine umfassende Regressionssuite auf, fokussiert auf die Bereiche des Systems, die tatsächlich Defekte produzieren.
- Etablieren Sie eine Richtlinie, dass kein Code gemergt wird, ohne alle bestehenden Regressionstests zu bestehen, und machen Sie Testfehlschläge zu einer blockierenden Bedingung im Review-Prozess. Diese Richtlinie fördert direkt häufigere Code-Einreichungen, weil Entwickler schnelles, automatisiertes Feedback erhalten, ob ihre Änderungen etwas brechen, was die Angst beseitigt, die reduzierte Einreichungshäufigkeit antreibt.
- Implementieren Sie für Legacy-Systeme, die sich schnell ändern, Regressionstests auf API-Vertragsebene zusätzlich zu Unit- und Integrationsebenen. API-Ebenen-Regressionstests erfassen brechende Änderungen, die Konsumenten betreffen, ohne von der internen Implementierung abzuhängen, die sich möglicherweise schnell weiterentwickelt.
- Verfolgen Sie das Wachstum der Regressionstest-Abdeckung über die Zeit als Kennzahl, die Fortschritt zu sichererer Modifizierbarkeit demonstriert. Berichten Sie Abdeckung in Begriffen abgedeckter kritischer Arbeitsabläufe statt Codezeilen, weil Stakeholder „wir können jetzt den Zahlungsablauf sicher modifizieren" besser verstehen als „wir haben 45 % Zeilenabdeckung erreicht".
- Reservieren Sie in jedem Sprint dedizierte Zeit zum Schreiben von Regressionstests rund um Code, der bald modifiziert werden soll, und behandeln Sie Testerstellung als Voraussetzung für die Modifikation statt als nachträglichen Gedanken. Dieser Ansatz stellt sicher, dass Testabdeckung dort wächst, wo sie am meisten gebraucht wird, angetrieben von tatsächlicher Entwicklungsaktivität statt abstrakten Abdeckungszielen.

## Tradeoffs ⇄

> Regressionstests verwandeln das unsichtbare Risiko der Legacy-System-Modifikation in sichtbare, handhabbare Verifikation, was dem Team erlaubt, von ängstlicher Vermeidung zu selbstbewusster Verbesserung überzugehen, erfordern aber anhaltende Investition in Testerstellung und -pflege.

**Vorteile:**

- Adressiert direkt Wartungslähmung, indem das Vertrauen geliefert wird, Code zu modifizieren, den das Team nicht anzufassen wagte, und den Kreislauf durchbricht, in dem die Angst, Dinge zu brechen, notwendige Verbesserungen verhindert.
- Reduziert übertriebene Schätzungen für kleine Änderungen, indem Entwicklern erlaubt wird, die Auswirkung von Modifikationen schnell zu verifizieren, statt umfangreiche manuelle Tests und Risikoanalyse vor jeder Änderung zu erfordern.
- Verringert steigende Fehlerzahlen, indem Regressionen erfasst werden, bevor sie Produktion erreichen, und verhindert den sich verstärkenden Effekt, bei dem jede Veröffentlichung neue Defekte einführt, die Qualität und Nutzervertrauen untergraben.
- Ermöglicht häufigere Code-Einreichungen, indem schnelles automatisiertes Feedback geliefert wird, das die langsame, manuelle Verifikation ersetzt, die Entwickler davon abhält, inkrementelle Änderungen einzureichen.
- Reduziert erhöhte Entwicklungskosten, indem Fehler früh im Entwicklungszyklus erfasst werden, wenn sie günstiger zu beheben sind, statt sie in Produktion zu entdecken, wo Diagnose und Reparatur teuer sind.
- Baut die Erosion des Nutzervertrauens wieder auf, indem die Häufigkeit sichtbarer Regressionen reduziert wird, die das Vertrauen der Nutzer in die Zuverlässigkeit des Systems beschädigen.

**Kosten und Risiken:**

- Regressionstests für Legacy-Systeme mit eng gekoppelten Komponenten und ohne Dependency Injection zu schreiben ist echt schwierig und könnte erfordern, den Code umzugestalten, um ihn testbar zu machen, was ein Henne-Ei-Problem schafft.
- Nicht gepflegte Regressionstestsuiten werden brüchig und produzieren falsche Fehlschläge, was Entwickler darauf trainiert, Testergebnisse zu ignorieren statt ihnen zu vertrauen, was schlimmer ist, als gar keine Tests zu haben.
- Langsame Regressionstestsuiten, die Stunden zum Abschluss brauchen, können zu Engpässen werden, die Veröffentlichungen verzögern und Entwickler frustrieren, was Investition in Testinfrastruktur, Parallelisierung und selektive Testausführung erfordert.
- Characterization Tests, die fehlerhaftes Verhalten erfassen, erzeugen eine Spannung, wenn das Team später diese Fehler beheben möchte: Die Tests müssen zusammen mit den Fixes aktualisiert werden, was Verständnis erfordert, welche Test-Assertionen beabsichtigtes Verhalten und welche Fehler repräsentieren.
- Übermäßiges Vertrauen auf Ende-zu-Ende-Regressionstests ohne ergänzende Unit- und Integrationstests schafft eine Testsuite, die langsam läuft, teuer zu pflegen ist und bei Fehlschlägen schlechte diagnostische Informationen liefert.

## How It Could Be

> Die folgenden Szenarien veranschaulichen, wie Regressionstests selbstbewusste Modifikation von Legacy-Systemen ermöglichen, wo Angst, Dinge zu brechen, zuvor Verbesserungsanstrengungen lähmte.

Eine Regierungsbehörde betreibt ein Legacy-Steuerberechnungssystem, das jährlich Millionen von Steuererklärungen verarbeitet. Das System hat keine automatisierten Tests, und das Entwicklungsteam konnte die Steuerregel-Engine zwei Jahre lang nicht aktualisieren, weil jeder vorherige Modifikationsversuch falsche Berechnungen verursachte, die erst entdeckt wurden, nachdem Erklärungen verarbeitet worden waren. Das Team beginnt damit, Characterization Tests zu schreiben, die die Ausgabe der bestehenden Regel-Engine für eine repräsentative Menge von 5.000 historischen Steuererklärungen erfassen, alle wichtigen Erklärungskategorien und Grenzfälle abdeckend. Diese Tests behaupten nicht, dass die Berechnungen korrekt sind — sie behaupten, dass die Ergebnisse der Ausgabe des aktuellen Systems entsprechen. Mit diesem Sicherheitsnetz modifiziert das Team die Regel-Engine, um neue Steuergesetzgebung zu unterstützen, wobei die Characterization Tests nach jeder Änderung ausgeführt werden, um zu verifizieren, dass nur die beabsichtigten Berechnungen sich geändert haben. Das zwei Jahre lang blockierte Update wird in sechs Wochen abgeschlossen, mit null Berechnungsfehlern nach der Bereitstellung. Die Characterization-Test-Suite wird zu einem dauerhaften Vermögenswert, der jährliche Steuerregel-Updates mit Vertrauen ermöglicht.

Das Entwicklungsteam eines SaaS-Unternehmens hat erlebt, dass ihre Code-Einreichungshäufigkeit von täglichen Pull Requests auf wöchentliche Batches gefallen ist, weil Entwickler Angst haben, dass ihre Änderungen die Anwendung brechen und einen Notfall-Produktions-Fix auslösen. Die Analyse zeigt, dass die vorherigen fünf Produktionsvorfälle alle Regressionen waren — Features, die zuvor korrekt funktionierten, aber durch scheinbar unabhängige Änderungen kaputtgingen. Das Team implementiert eine Regressionstest-Strategie, fokussiert auf die spezifischen Integrationspunkte, an denen Regressionen historisch auftraten. Für jede vergangene Regression schreiben sie einen Test, der sie erfasst hätte, und sie fügen Regressionstests für jeden neuen Bugfix hinzu, bevor sie den Fix mergen. Innerhalb von drei Monaten deckt die Regressionstest-Suite 85 % der Integrationspunkte ab, die historisch Fehlschläge produzieren. Das Entwicklervertrauen steigt messbar: Code-Einreichungen kehren zu einem täglichen Rhythmus zurück, Pull-Request-Größen schrumpfen von durchschnittlich 400 Zeilen auf 80 Zeilen, und die Code-Review-Qualität verbessert sich, weil Reviewer sich auf Design konzentrieren können, statt sich über versteckte Brüche zu sorgen. Die Anzahl der Produktionsregressionen fällt von fünf pro Quartal auf null über zwei aufeinanderfolgende Quartale.
