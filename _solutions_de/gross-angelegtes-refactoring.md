---
title: Groß angelegtes Refactoring
description: Durchführung einer verhaltenserhaltenden Änderung über viele Module
  oder Repositories in nachverfolgten Batches mit benanntem Verantwortlichen, sodass
  umfassende Refactorings tatsächlich abgeschlossen werden.
category:
- Process
- Code
- Team
problems:
- technology-stack-fragmentation
- inconsistent-execution
- mixed-coding-styles
- code-duplication
- obsolete-technologies
- dependency-version-conflicts
- shared-dependencies
- high-technical-debt
- inconsistent-naming-conventions
- incomplete-projects
- organizational-structure-mismatch
- undefined-code-style-guidelines
- api-versioning-conflicts
- legacy-api-versioning-nightmare
- maintenance-paralysis
- over-reliance-on-utility-classes
- refactoring-avoidance
- strangler-fig-pattern-failures
- excessive-customization
- core-modification-of-standard-software
layout: solution
lang: de
en_slug: large-scale-refactoring
related_solutions:
- slug: small-change-batches
  similarity: 0.75
- slug: automated-code-migration
  similarity: 0.75
- slug: incremental-refactoring
  similarity: 0.75
- slug: mikado-method
  similarity: 0.75
- slug: preparatory-refactoring
  similarity: 0.7
- slug: code-review-process-reform
  similarity: 0.7
---

## Description

Groß angelegtes Refactoring ist die organisatorische Hälfte einer umfassenden verhaltenserhaltenden Änderung: wie eine einzelne Transformation über Dutzende von Modulen oder Repositories angewendet wird, die verschiedenen Teams gehören, bis zur Fertigstellung verfolgt und danach aufgeräumt wird. Die Tooling-Hälfte — ein Rezept, das die Transformation durchführt — ist meist der einfachere Teil. Der Grund, warum solche Änderungen scheitern, ist fast nie technisch. Es ist, dass die Änderung auf sechzig Prozent des Bestands angewendet wird, die verbleibenden vierzig Prozent Teams mit ihren eigenen Prioritäten gehören, niemand die Fertigstellung besitzt, und die Codebasis dauerhaft in zwei Zuständen verbleibt. Dieser Endzustand ist schlimmer, als nie begonnen zu haben, weil jetzt sowohl das alte als auch das neue Muster unbegrenzt unterstützt werden müssen. Der Prozess existiert, um Fertigstellung zum Standardfall statt zur Ausnahme zu machen.

## How to Apply ◆

> Eine Migration, die bei achtzig Prozent stockt, lässt die Organisation für immer zwei Idiome pflegen, was genau das Ergebnis ist, das der Prozess verhindern soll.

- **Geben Sie der Änderung einen benannten Eigentümer**, der dafür verantwortlich ist, dass sie zur Fertigstellung gelangt, nicht dafür, die Änderung vorzunehmen. Umfassende Änderungen mit verteilter Eigentümerschaft stocken zuverlässig, und das Stocken ist niemandes Problem, es zu bemerken.
- **Etablieren Sie zuerst den echten Umfang.** Durchsuchen Sie den gesamten Bestand nach dem Muster, bevor Sie beginnen — einschließlich Repositories, an die niemand denkt, generiertem Code und Konfiguration. Eine Migration, deren Umfang schrittweise entdeckt wird, wird falsch geschätzt und findet ihre schlimmsten Fälle zuletzt.
- **Pilotieren Sie an einem eigenen Modul** und messen Sie, was es tatsächlich brauchte, einschließlich des Reviews und der Überraschungen. Dies liefert eine verteidigbare Schätzung für den Rest und, wichtiger, produziert ein durchgearbeitetes Beispiel, das anderen Teams gezeigt werden kann.
- **Bevorzugen Sie eine Kompatibilitätsschicht gegenüber synchronisiertem Umschalten.** Lassen Sie die alte und neue Form koexistieren — ein veralteter Wrapper, der an die neue API delegiert —, sodass jedes Modul unabhängig migrieren kann. Zu verlangen, dass alles auf einmal umschaltet, liefert die Änderung dem langsamsten Team aus.
- **Bündeln Sie den Rollout**, statt sechzig Pull Requests auf einmal zu öffnen. Reviewer ignorieren eine Flut, und ein Bündel von fünf oder zehn hält das Review sinnvoll und die Merge-Konflikte handhabbar.
- **Verfolgen Sie den Fortschritt öffentlich** — eine einfache Liste von Modulen und ihrem Zustand. Sichtbarer Fortschritt ist es, was eine Änderung über die Monate trägt, die sie braucht, und eine unvollständige Liste ist es, was den Rest besprechbar statt vergessen macht.
- **Machen Sie die Änderung für das empfangende Team leicht.** Liefern Sie den Pull Request, das Rezept, die Testergebnisse und eine einzeilige Erklärung, warum. Eine Migration, die andere Teams um Arbeit bittet, wird sich im Tempo von deren Prioritäten bewegen, was langsamer ist als Ihres.
- **Verhindern Sie Regression, während Sie fortschreiten.** Sobald ein Modul migriert ist, sollte eine Lint-Regel, ein Ratchet oder eine Build-Prüfung verhindern, dass das alte Muster zurückkehrt — sonst verkommen die frühesten Migrationen, während die letzten noch im Gange sind.
- **Löschen Sie den alten Pfad zu einem erklärten Datum**, und behandeln Sie das als Teil der Änderung statt als Nachfolgearbeit. Das Entfernen der Kompatibilitätsschicht ist der Schritt, der wegfällt, und ihn zu überspringen bedeutet, dass die Änderung nie tatsächlich ihren Nutzen geliefert hat.
- **Berichten Sie den Rest ehrlich.** Manche Module werden legitim nicht migrieren — eingefrorene Systeme, Drittanbietercode, etwas zur Stilllegung geplant. Sie zu benennen schließt die Änderung ab, statt sie dauerhaft bei 94 Prozent zu belassen.

## Tradeoffs ⇄

> Ein nachverfolgter Prozess ist es, was umfassende Änderungen zur Fertigstellung bringt, auf Kosten von Koordinationsaufwand und einer Periode, in der zwei Idiome koexistieren.

**Vorteile:**

- Die Änderung wird tatsächlich abgeschlossen, was den Unterschied zwischen einer konsistenten Codebasis und einer ausmacht, die dauerhaft zwei Wege trägt, dasselbe zu tun.
- Teams sind nicht voneinander blockiert, da die Kompatibilitätsschicht jedem erlaubt, nach eigenem Zeitplan zu migrieren.
- Die sichtbare Verfolgung trägt die Dynamik über die Monate, die solche Änderungen brauchen, und macht die verbleibende Arbeit besprechbar statt vergessen.
- Regressionsverhinderung, angewendet während Module landen, bedeutet, dass die frühe Arbeit nicht verkommt, während die späte im Gange ist.
- Der Pilot produziert eine echte Schätzung und ein durchgearbeitetes Beispiel, was weit überzeugender für andere Teams ist als eine Anfrage und eine Begründung.

**Kosten und Risiken:**

- Koordination über Teams hinweg ist echter Overhead, und er landet beim Eigentümer statt verteilt zu werden.
- Die Kompatibilitätsschicht ist selbst technische Schulden, und wenn der Löschschritt übersprungen wird, hat die Organisation eine permanente Schicht hinzugefügt statt eine entfernt.
- Große umfassende Änderungen erzeugen Merge-Konflikte mit allem, was gerade in Arbeit ist, was eine Steuer auf jedes Team während des Rollouts darstellt.
- Änderungen, die Teams aufgezwungen werden, die keinen Nutzen sehen, erzeugen Groll, und deren Module werden diejenigen sein, die ein Jahr später noch ausstehen.
- Der Prozess kann auf Änderungen angewendet werden, die ihn nicht rechtfertigen, was eine nette Konsistenzverbesserung in ein monatelanges Programm verwandelt.

## How It Could Be

Eine Organisation mit rund 40 Diensten wollte sich auf einen einzigen HTTP-Client standardisieren, nachdem sie über ein Jahrzehnt vier angesammelt hatte. Zwei vorherige Versuche hatten etwa die Hälfte der Dienste erreicht und waren dann gestoppt. Der dritte Versuch wies einer Ingenieurin die Eigentümerschaft mit 30 Prozent ihrer Zeit zu. Sie durchsuchte zuerst den gesamten Bestand und fand 47 Dienste statt der 40 im Dienstregister, drei davon ohne identifiziertes eigentümerschaftliches Team. Sie pilotierte an einem Dienst, maß es auf etwa einen halben Tag einschließlich Review, und schrieb ein Rezept, das den mechanischen Teil handhabte. Statt Teams zu bitten zu migrieren, öffnete sie die Pull Requests selbst in Bündeln von sechs, jeder mit angehängten Testergebnissen und einer zweisätzigen Erklärung. Neununddreißig Dienste wurden innerhalb von elf Wochen migriert. Die drei nicht eigentümerschaftlich geregelten wurden eskaliert und wurden zu einer Eigentümerschaftsentscheidung. Fünf wurden legitim ausgeschlossen — zwei eingefroren, drei innerhalb des Jahres zur Stilllegung geplant — und wurden als solche gelistet.

Der Löschschritt war es, wo die vorherigen Versuche im Rückblick tatsächlich gescheitert waren. Beide hatten Kompatibilitäts-Wrapper an Ort und Stelle belassen, sodass die alten Clients im Abhängigkeitsbaum blieben, weiterhin Sicherheitspatches erhielten und weiterhin von neuem Code genutzt wurden, weil sie noch verfügbar waren. Der dritte Versuch setzte von Anfang an ein Entfernungsdatum und entfernte die drei abgelösten Clients an diesem Datum, was zwei Dienste ans Licht brachte, die während des Rollouts zum alten Muster zurückgefallen waren — gefangen genau deshalb, weil jemand noch hinschaute. Die Abhängigkeitszahl der Organisation sank um drei Bibliotheken, und der wiederkehrende Sicherheitspatch-Aufwand, der mit ihnen verbunden war, hörte auf.
