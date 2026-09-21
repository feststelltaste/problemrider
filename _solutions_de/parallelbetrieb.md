---
title: Parallelbetrieb
description: Alte und neue Implementierung nebeneinander mit echtem Traffic
  betreiben, ihre Ausgaben vergleichen und erst umschalten, wenn die
  Unterschiede verstanden sind.
category:
- Architecture
- Testing
- Operations
problems:
- strangler-fig-pattern-failures
- fear-of-breaking-changes
- legacy-business-logic-extraction-difficulty
- history-of-failed-changes
- data-migration-complexities
- data-migration-integrity-issues
- regression-bugs
- hidden-side-effects
- insufficient-testing
- legacy-code-without-tests
- second-system-effect
- high-defect-rate-in-production
- schema-evolution-paralysis
- maintenance-paralysis
- release-anxiety
- entity-attribute-value-overuse
- retention-obligations-block-change
- upgrade-blocked-by-customization
layout: solution
lang: de
en_slug: parallel-run
related_solutions:
- slug: blue-green-canary-deployments
  similarity: 0.7
- slug: feature-flags
  similarity: 0.7
- slug: strangler-fig-pattern
  similarity: 0.7
- slug: ci-cd-pipeline
  similarity: 0.7
- slug: large-scale-refactoring
  similarity: 0.65
- slug: test-coverage-strategy
  similarity: 0.65
---

## Description

Ein Parallelbetrieb führt die Ersatzimplementierung neben derjenigen aus, die sie ersetzen soll, speist beiden dieselben echten Eingaben und vergleicht ihre Ausgaben, während nur die Ergebnisse des Originals verwendet werden. Es ist die stärkste verfügbare Antwort auf die Frage, die die meisten Legacy-Ersetzungen blockiert: Woher wissen wir, dass sich die neue genauso verhält wie die alte? Tests beantworten das nur für Fälle, an die jemand gedacht hat, und die Fälle, die in einem jahrzehntealten System zählen, sind die, an die niemand gedacht hat — der Kundendatensatz mit einem Null-Wert in einem Feld, das seit 2004 verpflichtend war, der von einem Kunden verwendete Abrechnungstyp. Produktions-Traffic enthält diese Fälle, eine Testsuite nicht. Der Parallelbetrieb verwandelt die Umschaltung von einer Entscheidung, die auf der Stärke einer Testsuite getroffen wird, in eine, die auf der Stärke beobachteter Übereinstimmung über echte Daten getroffen wird.

## How to Apply ◆

> Der Wert der Technik ist proportional dazu, wie wenig Sie das Original verstehen, was sie genau dort am angemessensten macht, wo das Risiko am höchsten ist.

- **Leiten Sie echte Eingaben an beide Implementierungen weiter**, während Sie nur die Ausgabe des Originals verwenden. Der neue Pfad darf während der Vergleichsperiode keine Seiteneffekte haben: keine Schreibvorgänge in gemeinsame Tabellen, keine veröffentlichten Nachrichten, keine externen Aufrufe. Diese Isolation falsch zu machen ist der Hauptweg, wie ein Parallelbetrieb genau den Vorfall verursacht, den er verhindern sollte.
- **Vergleichen Sie Ausgaben automatisch und protokollieren Sie jeden Unterschied** mit genug Kontext, um ihn zu reproduzieren — die Eingabe, beide Ausgaben und einen Zeitstempel. Manueller Vergleich skaliert nicht über die ersten paar hundert Fälle hinaus, und die interessanten Unterschiede sind selten.
- **Kategorisieren Sie Unterschiede, statt sie zu zählen.** Tausend Diskrepanzen aus einer Rundungsregel sind ein Befund; drei Diskrepanzen aus drei unterschiedlichen Ursachen sind drei. Fortschritt wird an geschlossenen Kategorien gemessen, nicht an der Diskrepanzrate.
- Erwarten Sie, dass **manche Unterschiede Fehler im Original sind**, und entscheiden Sie bewusst für jeden: das alte Verhalten reproduzieren, weil Konsumenten davon abhängen, oder es beheben und diese Konsumenten informieren. Diese Entscheidung zu dokumentieren ist wichtig, weil die nächste Person sonst die bewusste Reproduktion eines Fehlers als Versehen liest.
- **Laufen Sie lange genug, um den Geschäftszyklus abzudecken.** Monatliche und quartalsweise Verarbeitungspfade erscheinen nicht innerhalb einer Woche. Für Finanz- und Abrechnungssysteme bedeutet dies üblicherweise mindestens einen vollständigen Monatsabschluss und oft einen Quartalsabschluss, bevor Übereinstimmung irgendetwas bedeutet.
- **Beobachten Sie die Kosten.** Die Verdoppelung der Berechnung ist für die meisten Request-Response-Arbeiten erschwinglich und kann für schwere Batch-Verarbeitung unerschwinglich sein. Wo das der Fall ist, samplen Sie — einen konsistenten Prozentsatz des Traffics, plus bewusstes Übersamplen seltener Eingabetypen, wo sich die Unterschiede konzentrieren.
- **Schalten Sie schrittweise um statt auf einmal**, wenn die Kategorien geschlossen sind: verschieben Sie einen kleinen Anteil des Traffics zur neuen Implementierung als Autorität, überwachen Sie und erhöhen Sie. Halten Sie den Vergleich in dieser Phase am Laufen, mit vertauschten Rollen.
- **Behalten Sie die Fähigkeit zum Zurücksetzen** bis lange nach der vollständigen Umschaltung. Die Unterschiede, die einen Parallelbetrieb überleben, sind diejenigen, die mit Frequenzen auftreten, die das Beobachtungsfenster nicht abgedeckt hat, und sie tauchen Wochen später auf.
- **Entfernen Sie die alte Implementierung bewusst**, zu einem geplanten Datum. Parallelbetriebe, die nie beendet werden, hinterlassen zwei zu wartende Implementierungen, was schlechter ist als die Situation vor Beginn der Migration.

## Tradeoffs ⇄

> Ein Parallelbetrieb liefert Nachweise für Äquivalenz, die keine andere Technik bietet, im Austausch gegen echte Infrastrukturarbeit, verdoppelte Verarbeitungskosten und einen längeren Zeitplan, bevor irgendein Nutzen realisiert wird.

**Vorteile:**

- Äquivalenz wird gegen echte Produktionsdaten demonstriert, einschließlich der seltenen Fälle, die den Großteil des Risikos und keine der Testabdeckung ausmachen.
- Die Umschaltentscheidung wird evidenzbasiert statt ein Vertrauensvorschuss, was üblicherweise das ist, was eine Ersetzung freischaltet, die an berechtigter Angst festgefahren war.
- Fehler im Original werden als Nebenprodukt entdeckt, häufig einschließlich solcher, die jahrelang still Schaden angerichtet haben.
- Das undokumentierte Verhalten des Originals wird konkret erfasst, was als die Spezifikation dient, die der Ersatz nie hatte.
- Vertrauen sammelt sich sichtbar über die Zeit an, was den Aufwand gegenüber Stakeholdern während der langen Periode verteidigbar macht, in der er keine für Nutzer sichtbare Veränderung produziert.

**Kosten und Risiken:**

- Alles wird zweimal berechnet, was für schwere Workloads teuer oder ohne Sampling nicht durchführbar sein kann.
- Der Bau der Routing-, Vergleichs- und Berichtsinfrastruktur ist echte Arbeit, die für sich genommen nichts liefert, und sie muss gebaut werden, bevor irgendwelche Vergleichsdaten existieren.
- Seitenwirkungen im Schattenpfad verursachen genau den Produktionsvorfall, den die Technik verhindern soll; Isolation muss verifiziert statt angenommen werden.
- Unterschiede können zahlreich genug sein, um demoralisierend zu wirken, und Teams senken manchmal ihren Standard für akzeptable Abweichung, um fertig zu werden.
- Die Wartung zweier Implementierungen während des Betriebs verdoppelt die Kosten jeder in der Zwischenzeit vorgenommenen Änderung, was Druck erzeugt, Features einzufrieren, und Druck, den Betrieb zu verkürzen.

## How It Could Be

Eine Bank ersetzte eine Gebührenberechnungs-Engine, die über achtzehn Jahre Regeln angesammelt hatte, ohne Spezifikation und ohne überlebenden Autor. Das Team baute den Ersatz aus dem Code, dann betrieb es ihn elf Wochen lang im Schatten gegen Produktions-Traffic. Die erste Woche produzierte Diskrepanzen bei 6,2 Prozent der Transaktionen, die sich in neun Kategorien auflösten. Sieben waren Fehler in der neuen Implementierung. Eine war ein Rundungsunterschied, der sich als Fehler im Original herausstellte, vorhanden seit 2013, der systematisch ein Produkt um Bruchteile eines Cents unterberechnet hatte — der akkumulierte Betrag war erheblich genug, um eine Offenlegung zu erfordern. Die neunte Kategorie erschien nur zum Monatsende und betraf einen Gebührenerlass, der auf Konten angewendet wurde, die mitten im Zyklus geschlossen wurden, eine Regel, die in keinem Dokument existierte und die die Fachseite als beabsichtigt bestätigte. Die Umschaltung erfolgte bei 0,02 Prozent Abweichung, alle in bewusst akzeptierten Kategorien, und produzierte keine Vorfälle.

Ein zweites Team wandte die Technik auf eine Datenmigration statt auf eine Berechnung an. Statt eines einzelnen Umschalt-Wochenendes schrieben sie vier Monate lang jede Datensatzänderung sowohl in das alte als auch das neue Schema, mit einem nächtlichen Job, der die beiden verglich und Abweichungen nach Tabelle meldete. Der Vergleich brachte drei Problemklassen zutage, die kein Testlauf gefunden hatte: einen Zeichenkodierungsunterschied bei Namen mit diakritischen Zeichen, eine Zeitzonenannahme in einer Datumsspalte, und eine Reihe von Datensätzen, die die Anwendungsschicht des alten Systems erzeugen konnte, die aber die eigenen Schemabeschränkungen technisch verboten. Alle drei wären während eines Umschalt-Wochenendes entdeckt worden, um drei Uhr morgens, mit einer Rollback-Frist. Stattdessen wurde jedes während normaler Arbeitszeit behoben, und die schließliche Umschaltung bestand darin, zu ändern, welches Schema die Anwendung las.
