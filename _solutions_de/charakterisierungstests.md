---
title: Charakterisierungstests
description: Festhalten dessen, was Legacy-Code aktuell tut — korrekt oder nicht
  — als ausführbare Tests, was ein Sicherheitsnetz für die Änderung von Code mit
  unbekanntem beabsichtigtem Verhalten schafft.
category:
- Testing
- Code
problems:
- legacy-code-without-tests
- poor-test-coverage
- outdated-tests
- testing-complexity
- fear-of-breaking-changes
- fear-of-change
- delayed-bug-fixes
- partial-bug-fixes
- increased-manual-testing-effort
- defensive-coding-practices
- legacy-business-logic-extraction-difficulty
- difficult-to-test-code
- maintenance-paralysis
- regression-bugs
- flaky-tests
- strangler-fig-pattern-failures
- global-state-and-side-effects
- increased-bug-count
- refactoring-avoidance
- test-debt
- cache-invalidation-problems
- hidden-side-effects
- history-of-failed-changes
- increasing-brittleness
- monolithic-functions-and-classes
- brittle-codebase
- entity-attribute-value-overuse
- core-modification-of-standard-software
- upgrade-blocked-by-customization
layout: solution
lang: de
en_slug: characterization-tests
related_solutions:
- slug: test-coverage-strategy
  similarity: 0.85
- slug: functional-tests
  similarity: 0.75
- slug: regression-testing
  similarity: 0.75
- slug: exploratory-testing
  similarity: 0.75
- slug: automated-tests
  similarity: 0.75
- slug: static-analysis-and-linting
  similarity: 0.7
---

## Description

Ein Charakterisierungstest zeichnet auf, was ein Stück Code tatsächlich tut, statt was es tun soll. Man ruft den Code mit einer Reihe von Eingaben auf, beobachtet die Ausgabe und behauptet, dass die Ausgabe gleich bleibt. Dies kodiert bewusst bestehende Bugs als erwartetes Verhalten, was anfangs beunruhigend ist und genau der Punkt ist: Für Legacy-Code ohne Spezifikation, ohne Dokumentation und ohne überlebenden Autor ist das aktuelle Verhalten die einzige Spezifikation, die existiert, und nachgelagerte Systeme haben sich seit Jahren darauf verlassen — einschließlich seiner Defekte. Charakterisierungstests sind kein Ersatz für ordentliche Tests; sie sind das Gerüst, das es sicher macht, an Code lange genug zu arbeiten, um schließlich ordentliche zu schreiben. Ihre Funktion ist es, „ich weiß nicht, was das tut" von einem Grund, den Code nicht anzufassen, in eine dokumentierte, ausführbare Tatsache zu verwandeln.

## How to Apply ◆

> Das typische Ziel ist ein Modul, das jeder vermeidet, weil niemand weiß, was von seinen Eigenheiten abhängt — was genau das Modul ist, das am dringendsten geändert werden muss.

- **Finden Sie die kleinste Naht**, durch die der Code aufgerufen werden kann. Oft ist es überhaupt kein Unit-Level-Einstiegspunkt, sondern ein HTTP-Endpunkt, ein Batch-Job oder eine Datenbankprozedur. Testen Sie auf welcher Ebene auch immer heute erreichbar ist; ein grober Test, der existiert, schlägt einen feingranularen Test, der zunächst sechs Wochen Refaktorierung erfordert.
- **Schreiben Sie einen Test, der etwas behauptet, das Sie als falsch wissen**, führen Sie ihn aus und lassen Sie ihn fehlschlagen. Die Fehlermeldung sagt Ihnen den tatsächlichen Wert. Fügen Sie diesen Wert in die Assertion ein. Dies klingt roh und ist der schnellste verlässliche Weg, Verhalten zu charakterisieren, das Sie durch Lesen nicht vorhersagen können.
- Nutzen Sie **Approval Testing** für Ausgaben, die zu groß oder komplex sind, um Feld für Feld zu behaupten: Serialisieren Sie die gesamte Ausgabe in eine Datei, überprüfen Sie sie einmal von Hand und committen Sie sie als genehmigte Baseline. Nachfolgende Läufe diffen dagegen. Dies ist der praktische Ansatz für Berichtsgeneratoren, Dokumentenproduzenten und Nachrichtentransformatoren.
- **Generieren Sie Eingabeabdeckung systematisch** statt intuitiv. Grenzwerte, Nulls, leere Sammlungen und, wo verfügbar, eine Stichprobe echter Produktionseingaben werden Pfade ausüben, die handgeschriebene Beispiele verpassen. Eine Reihe echter Produktionsanfragen aufzuzeichnen und erneut abzuspielen ist oft der wertvollste Einzelschritt.
- Nutzen Sie **Abdeckungsmessung als Leitfaden, nicht als Ziel**, während der Charakterisierungsphase. Ihr Zweck hier ist es, zu offenbaren, welche Zweige Ihre Eingaben noch nicht erreicht haben, sodass Sie Eingaben konstruieren können, die es tun. Ein Zweig, der von keinem Charakterisierungstest je ausgeübt wurde, ist ein Zweig, den Sie gleich blind ändern.
- **Markieren Sie die Tests explizit als Charakterisierungstests** — eine Namenskonvention, eine Annotation, ein separates Verzeichnis. Spätere Leser müssen erkennen können, dass diese Assertions beobachtetes Verhalten beschreiben, nicht beabsichtigtes, sonst wird schließlich jemand einen Test „reparieren", der einen Bug dokumentiert, auf den sich ein nachgelagertes System verlässt.
- Wenn ein Charakterisierungstest **später einen tatsächlichen Bug offenbart**, entscheiden Sie bewusst und dokumentieren Sie die Entscheidung: beheben Sie ihn und aktualisieren Sie den Test mit einer Notiz, die die Änderung erklärt, oder behalten Sie das Verhalten bei und dokumentieren Sie, warum es bewahrt werden muss. Ändern Sie keines davon still.
- **Konvertieren Sie schrittweise zu Spezifikationstests.** Während das beabsichtigte Verhalten eines Bereichs verstanden wird, ersetzen Sie Charakterisierungs-Assertions durch Tests, die die Anforderung formulieren, und löschen Sie die redundanten Baselines. Die Charakterisierungs-Suite sollte über die Lebensdauer einer Modernisierungsbemühung schrumpfen.
- Akzeptieren Sie, dass die Suite **hässlich und repetitiv** sein wird. Charakterisierungstests sind Tooling für eine Übergangsperiode, keine Codebasis, auf die man stolz sein soll, und Zeit, die aufgewendet wird, um sie elegant zu machen, ist üblicherweise besser genutzt, um ihre Abdeckung zu erweitern.

## Tradeoffs ⇄

> Charakterisierungstests bieten schnell und günstig ein Sicherheitsnetz, zum Preis einer Test-Suite, die Defekte als Anforderungen dokumentiert und später aktiv abgebaut werden muss.

**Vorteile:**

- Refaktorierung wird in Code möglich, der zuvor nicht sicher angefasst werden konnte, was die Voraussetzung für im Wesentlichen jede andere Verbesserung an einem Legacy-Modul ist.
- Bestehendes Verhalten — einschließlich undokumentierten Verhaltens, auf das sich Konsumenten verlassen — wird durch Änderungen bewahrt, was das Hauptrisiko in Legacy-Modifikation ist und ohne Tests am schwersten zu beurteilen.
- Undokumentierte Geschäftsregeln werden sichtbar und lesbar, da die Tests eine ausführbare Beschreibung dessen sind, was das System tatsächlich tut. Sie werden häufig zur ersten echten Dokumentation eines Moduls.
- Bugs werden als Nebeneffekt entdeckt: das Schreiben von Eingaben, die jeden Zweig regelmäßig ausüben, bringt Verhalten ans Licht, das niemand beabsichtigt hat und niemand bemerkt hatte.
- Die Tests können von jemandem geschrieben werden, der die Domäne nicht versteht, was wichtig ist, wenn die ursprünglichen Entwickler weg sind und kein Domänenexperte verfügbar ist.

**Kosten und Risiken:**

- Die Suite kodiert aktuelle Bugs als erwartetes Verhalten. Ohne klare Markierung werden spätere Entwickler diese Assertions als Anforderungen behandeln und Defekte unbegrenzt bewahren.
- Charakterisierungstests sind konstruktionsbedingt brüchig: Sie schlagen bei jeder Verhaltensänderung fehl, einschließlich beabsichtigter, was Rauschen produziert und dazu führen kann, dass die Suite ignoriert oder ohne Review massenhaft aktualisiert wird.
- Grobgranulare Tests durch die äußerste Naht sind langsam, und eine langsame Suite wird seltener ausgeführt, was das Sicherheitsnetz genau dann erodiert, wenn Änderungen am schnellsten vorgenommen werden.
- Approval-Baselines können gedankenlos aktualisiert werden. Ein Workflow, bei dem das Regenerieren der Baseline einfacher ist als das Verstehen des Diffs, besiegt den gesamten Mechanismus, sodass Baseline-Änderungen wie Code überprüft werden müssen.
- Die Tests bieten keine Anleitung darüber, was der Code tun sollte, sodass sie gegen Regression schützen, während sie keine Hilfe bei der Entscheidung bieten, was gebaut werden soll.

## How It Could Be

Ein Team musste Steuerberechnungslogik in einem Gehaltsabrechnungssystem ändern: 4.000 Zeilen verschachtelter Bedingungen, keine Tests, und der ursprüngliche Entwickler war sechs Jahre zuvor in den Ruhestand gegangen. Statt ihn zu lesen, schrieben sie einen Harness, der die Berechnung gegen 12.000 anonymisierte historische Gehaltsabrechnungsdatensätze ausführte und die Ergebnisse als genehmigte Baseline speicherte. Der Bau des Harness dauerte vier Tage. Er offenbarte sofort, dass elf Datensätze bei wiederholten Läufen unterschiedliche Ergebnisse produzierten — eine Abhängigkeit von der Systemuhr, von der niemand gewusst hatte, und die seit Jahren still gelegentlich falsche Abzüge produzierte. Mit der Baseline vorhanden refaktorierte das Team das Modul über fünf Wochen, wobei nach jeder Änderung der vollständige Vergleich ausgeführt wurde. Zwei der Zwischenänderungen produzierten unerwartete Diffs und wurden innerhalb von Minuten zurückgesetzt.

Ein anderes Team nutzte Charakterisierungstests, um eine Entscheidung zu treffen, statt eine Refaktorierung zu ermöglichen. Sie bewerteten, ob ein Rechnungsgenerator durch ein Anbieterprodukt ersetzt werden könnte. Sie charakterisierten die Ausgabe des bestehenden Generators über 800 Beispielrechnungen, führten dann dieselben Eingaben durch das Kandidatenprodukt und diffeten. Der Vergleich fand 43 systematische Unterschiede, von denen sich 6 als rechtliche Anforderungen spezifisch für zwei Rechtsgebiete herausstellten, die das Anbieterprodukt nicht unterstützte. Dieser Befund, produziert in zwei Wochen, verhinderte eine Beschaffungsentscheidung, die sich sonst ungefähr neun Monate in ein Integrationsprojekt hinein als unpraktikabel herausgestellt hätte.
