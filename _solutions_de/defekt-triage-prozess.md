---
title: Defekt-Triage-Prozess
description: Bewertung jedes gemeldeten Defekts anhand festgelegter Kriterien in
  regelmäßigem Takt, Klassifizierung seiner Ursache und Nutzung der angesammelten
  Klassifizierung, um Kategorien statt Einzelfälle zu beheben.
category:
- Process
- Code
- Testing
problems:
- partial-bug-fixes
- delayed-bug-fixes
- increased-bug-count
- quality-degradation
- high-defect-rate-in-production
- constant-firefighting
- delayed-issue-resolution
- regression-bugs
- quality-compromises
- brittle-codebase
- reduced-feature-quality
- workaround-culture
- avoidance-behaviors
- blame-culture
- increased-risk-of-bugs
- increasing-brittleness
- negative-brand-perception
- user-trust-erosion
layout: solution
lang: de
en_slug: defect-triage-process
related_solutions:
- slug: explicit-prioritization-framework
  similarity: 0.65
- slug: workaround-registry
  similarity: 0.65
- slug: code-hotspot-analysis
  similarity: 0.65
- slug: blameless-postmortems
  similarity: 0.65
- slug: debt-classification
  similarity: 0.65
- slug: debt-accrual-analysis
  similarity: 0.65
---

## Description

Ein Defekt-Triage-Prozess ist eine regelmäßige, kurze Überprüfung, bei der neu gemeldete Defekte anhand festgelegter Kriterien bewertet werden — Schweregrad, betroffene Nutzer, ob ein Workaround existiert —, einen Eigentümer und eine Priorität zugewiesen bekommen und nach zugrunde liegender Ursache klassifiziert werden. Die Klassifizierung ist der Teil, den Teams überspringen, und der Teil, der am wichtigsten ist. Defekte einzeln zu behandeln, in der Reihenfolge, in der sie eintreffen, oder in der Reihenfolge, in der über sie geschrien wird, bedeutet, für immer Symptome zu beheben: Dieselbe Defektklasse tritt wieder auf, weil niemand über Instanzen hinweg geschaut hat, um zu sehen, dass dreißig von ihnen eine Ursache teilen. Legacy-Systeme erzeugen Defekte schneller, als irgendein Team sie einzeln beheben kann, sodass die einzig handhabbare Strategie ist, Kategorien zu beheben. Triage ist der Mechanismus, der einen Strom einzelner Meldungen in die Daten verwandelt, die nötig sind, um diese Kategorien zu identifizieren.

## How to Apply ◆

> In einem System, das mehr Defekte produziert, als das Team beheben kann, wird die Entscheidung, was nicht behoben wird, entweder explizit durch Triage getroffen oder implizit von wem auch immer am lautesten klagt.

- Halten Sie Triage in einem **festen, häufigen Takt** ab — zweimal pro Woche für ein hochvolumiges System — und halten Sie sie kurz. Ein langes, seltenes Triage-Meeting häuft einen Rückstau an, der zu groß ist, um ordentlich bewertet zu werden, sodass Posten mit einer Vermutung durchgewunken werden.
- Nutzen Sie **schriftliche Schweregradkriterien** statt Ermessen im Moment. Datenkorruption, Sicherheitsexposition, blockierter Geschäftsprozess, verschlechterte Erfahrung und kosmetisch reichen aus. Ohne schriftliche Kriterien folgt der Schweregrad dem, wer es gemeldet hat.
- Erfassen Sie, **ob ein Workaround existiert und was er kostet**. Ein Defekt hoher Schwere mit günstigem Workaround kann vernünftigerweise hinter einem mittleren ohne einen warten; ohne dieses Feld wird die Prioritätsentscheidung allein auf Basis des Schweregrads getroffen und ist häufig falsch.
- **Klassifizieren Sie die Ursache, nicht nur das Symptom** — fehlende Validierung, unbehandelter Null-Wert, Race Condition, Konfigurationsfehler, missverstandene Anforderung, Regression aus einer anderen Änderung. Nutzen Sie eine kleine feste Taxonomie von acht bis zwölf Kategorien, damit die Daten vergleichbar bleiben.
- **Überprüfen Sie die Klassifizierungsverteilung vierteljährlich.** Hier liegt der Wert. Eine dominante Kategorie ist ein systemischer Befund, und sie zu beheben adressiert Defekte, die noch nicht gemeldet wurden. Einzelne Triage-Entscheidungen zählen weit weniger als dieses Aggregat.
- Weisen Sie **jedem akzeptierten Defekt bei Triage einen Eigentümer** zu, nicht später. Defekte ohne Eigentümer altern unbegrenzt, und das Alter eines Defekts ist die Kennzahl, die am besten vorhersagt, ob überhaupt etwas je behoben wird.
- **Entscheiden Sie explizit, nicht zu beheben**, wo das die Antwort ist, und erfassen Sie warum. Ein unbehobener Defekt, der für immer offen liegt, ist schlimmer als ein geschlossener mit angegebenem Grund, weil er die Daten verschmutzt und dem Melder falsche Hoffnung gibt.
- Verlangen Sie, dass eine Korrektur die **Ursache statt der Instanz** adressiert, oder dass die partielle Natur der Korrektur erfasst wird. Partielle Korrekturen sind unter Zeitdruck manchmal richtig; sie werden nur zum Problem, wenn niemand vermerkt, dass die zugrunde liegende Ursache bestehen bleibt.
- **Verfolgen Sie Regressionen separat.** Ein durch eine kürzliche Änderung eingeführter Defekt ist ein anderes Signal als einer, der jahrelang latent war, und eine steigende Regressionsrate verweist auf die Testsuite statt auf den Code.
- Speisen Sie die Klassifizierungsdaten in das **Verbesserungsbudget und die Hotspot-Analyse** ein. Die Kategorien, die die Verteilung dominieren, sind der beste verfügbare Beleg dafür, worin investiert werden soll.

## Tradeoffs ⇄

> Triage macht Priorisierung explizit und produziert die Daten, die nötig sind, um Ursachen zu beheben, auf Kosten wiederkehrender Besprechungszeit und einer Klassifizierungsdisziplin, die leicht verfällt.

**Vorteile:**

- Priorisierung bewegt sich von sozialem Druck zu festgelegten Kriterien, was verteidigungsfähiger ist und bessere Ergebnisse für Nutzer erzeugt, die nicht gut darin sind zu eskalieren.
- Ursachenklassifizierung identifiziert systemische Probleme, die in einzelnen Meldungen unsichtbar sind, was der einzige Weg ist, das Defektvolumen zu reduzieren, statt nur mitzuhalten.
- Defekte bekommen sofort Eigentümer, was der stärkste einzelne Prädiktor dafür ist, ob sie behoben werden.
- Bewusste Nicht-Behebungsentscheidungen räumen den Rückstau von Posten, die nie angegangen worden wären, und machen die verbleibende Liste bedeutungsvoll.
- Regressionstrends geben frühe Warnung über Testabdeckungserosion, meist deutlich bevor sie sich als Produktionsvorfall zeigt.

**Kosten und Risiken:**

- Wiederkehrende Besprechungszeit für mehrere Personen, was echte Kosten sind und oft das Erste, was fallengelassen wird, wenn das Team beschäftigt ist — genau dann, wenn das Defektvolumen am höchsten ist.
- Klassifizierung verschlechtert sich hin zu welcher Kategorie auch immer am leichtesten auszuwählen ist, und sobald die Daten unzuverlässig sind, wird die Aggregatanalyse irreführend statt bloß nutzlos.
- Schriftliche Kriterien werden ausgetrickst. Melder lernen, welche Wörter hohen Schweregrad erzeugen, und die Kriterien brauchen gelegentliche Neukalibrierung.
- Triage kann zum Engpass werden, wenn das Meeting der einzige Pfad zu einer Entscheidung ist, was echt dringende Posten verzögert, die es umgehen sollten.
- Explizit abzulehnen, Defekte zu beheben, ist politisch unangenehm und kann Beziehungen zu den Menschen schädigen, die sie gemeldet haben, selbst wenn es die richtige Entscheidung ist.

## How It Could Be

Ein Team, das ein Einzelhandels-Kassensystem pflegte, erhielt 60 bis 90 Defektmeldungen pro Monat und behob etwa 40, ausgewählt durch eine Mischung aus Schweregrad und wer fragte. Sie führten zweimal wöchentliche Triage mit fünf schriftlichen Schweregradstufen, einem Workaround-Feld und einer neunkategorigen Ursachentaxonomie ein. Die erste vierteljährliche Überprüfung der Klassifizierungsdaten zeigte, dass 38 Prozent aller Defekte in eine Kategorie fielen: unbehandelte Randfälle in der Datums- und Zeitbehandlung, verstreut über elf unterschiedliche Module. Niemand hatte das gesehen, weil jede Instanz lokal von wem auch immer sie aufgriff behoben worden war. Das Team baute ein gemeinsames Datumsbehandlungsmodul und migrierte die elf Aufrufstellen über zwei Monate. Meldungen in dieser Kategorie sanken von durchschnittlich 26 im Monat auf 3.

Die explizite Nicht-Behebungsentscheidung veränderte den Rückstau mehr als jede Korrektur. Triage arbeitete über sechs Wochen 340 offene Defekte durch und schloss 190 davon als bewusst nicht behoben, jeder mit angegebenem Grund: überholt, nicht mehr reproduzierbar, betrifft ein zur Entfernung geplantes Feature oder als die Kosten nicht wert beurteilt. Neunzehn wurden von ihren Meldern wieder geöffnet, und vier davon erwiesen sich als echt wichtig und wurden behoben. Die verbleibenden 150 offenen Defekte waren zum ersten Mal eine Liste, die das Team tatsächlich durcharbeiten wollte, was bedeutete, dass das Alter des ältesten offenen Defekts zu einer berichtenswerten Kennzahl wurde statt zu einer Quelle der Verlegenheit.
