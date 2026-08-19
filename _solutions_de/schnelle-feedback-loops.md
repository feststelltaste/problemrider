---
title: Schnelle Feedback-Loops
description: Die Zeit von einer Änderung bis zum Wissen, ob sie funktioniert hat,
  als primäre Engineering-Metrik behandeln und angehen, was immer sie dominiert.
category:
- Code
- Process
- Testing
problems:
- long-build-and-test-times
- slow-development-velocity
- slow-feature-development
- inefficient-development-environment
- development-disruption
- reduced-code-submission-frequency
- extended-cycle-times
- flaky-tests
- context-switching-overhead
- tool-limitations
- increased-manual-work
- reduced-individual-productivity
- long-release-cycles
- automated-tooling-ineffectiveness
- delayed-bug-fixes
- excessive-logging
- extended-review-cycles
- fear-of-failure
- mental-fatigue
- reduced-review-participation
- review-bottlenecks
- review-process-avoidance
- testing-environment-fragility
layout: solution
lang: de
en_slug: fast-feedback-loops
related_solutions:
- slug: development-environment-optimization
  similarity: 0.75
- slug: small-change-batches
  similarity: 0.7
- slug: delivery-performance-metrics
  similarity: 0.7
- slug: regression-testing
  similarity: 0.7
- slug: development-workflow-automation
  similarity: 0.65
- slug: quality-ratchet
  similarity: 0.65
---

## Description

Ein schneller Feedback-Loop ist das Intervall zwischen dem Vornehmen einer Änderung und dem Herausfinden, ob sie das Beabsichtigte bewirkt hat. Er ist der Multiplikator für alles andere, was ein Team tut, weil jede Praxis, die von Iteration abhängt — Testen, Refactoring, kleine Batches, Debugging —, im Verhältnis zur Länge dieses Intervalls verkommt. Ein vierzigminütiger Build macht ein Team nicht vierzig Minuten langsamer; er verändert dessen Verhalten. Man bündelt Änderungen, um die Kosten zu vermeiden, hört auf, Tests lokal auszuführen, wechselt beim Warten den Kontext und verliert den Faden, und debuggt durch Überlegung statt durch Experiment. In Legacy-Systemen ist der Loop meist lang, und niemand hat ihn gemessen, weil er über Jahre minutenweise gewachsen ist und jeder Zuwachs zu klein war, um darauf zu reagieren. Ihn zu messen und anzugehen ist häufig die Engineering-Arbeit mit dem höchsten verfügbaren Ertrag, und sie wird fast immer unterschätzt, weil sie kein Feature liefert.

## How to Apply ◆

> Ein Team, das keinen aussagekräftigen Test in unter einer Minute ausführen kann, wird testgetriebene Entwicklung, inkrementelles Refactoring oder kleine Batches nicht praktizieren, egal wie sehr es dazu ermutigt wird.

- **Messen Sie die Loops, die Sie tatsächlich haben**, getrennt: Zeit zum Kompilieren, zum Ausführen eines Tests, zum Ausführen der schnellen Suite, zum Ausführen von allem, um eine funktionierende lokale Umgebung zu erhalten, und um irgendwohin zu deployen, wo man nachsehen kann. Teams entdecken regelmäßig, dass die dominante Kostenquelle nicht die ist, über die sie sich beschweren.
- **Greifen Sie zuerst den innersten Loop an.** Der Zyklus Kompilieren-und-einen-Test-ausführen wird hunderte Male pro Tag genutzt; die volle Pipeline wenige Male. Eine Verbesserung im inneren Loop ist weit mehr wert als dieselbe Verbesserung weiter außen, auch wenn die äußere Zahl größer und sichtbarer ist.
- **Teilen Sie die Testsuite nach Geschwindigkeit**, nicht nach Typ. Eine schnelle Suite, die bei jeder Änderung in unter zwei Minuten läuft, und eine langsamere, die beim Merge läuft, gibt den größten Teil der Sicherheit mit einem nutzbaren Loop. Die Aufteilung lohnt sich, auch wenn sie unvollkommen ist.
- **Beseitigen Sie Flakiness aggressiv.** Eine Suite, die zufällig fehlschlägt, ist kein langsamer Feedback-Loop, sondern ein kaputter: Entwickler hören auf, Fehlschlägen zu glauben, was den Wert, sie überhaupt auszuführen, zunichtemacht. Quarantänieren Sie flakige Tests sofort und beheben oder löschen Sie sie mit einer Frist.
- **Machen Sie die lokale Umgebung reproduzierbar und schnell erstellbar.** Wo ein Entwickler einen Tag braucht, um ein funktionierendes Setup zu bekommen, oder sich eine Integrationsumgebung mit fünf Kollegen teilt, ist das der Loop — und Containerisierung oder skriptgesteuerte Bereitstellung adressiert das meist schneller als jede Testoptimierung.
- **Entfernen Sie Arbeit aus dem Loop, statt sie schneller zu machen.** Einmal erstellte und wiederverwendete Testdaten, nicht neu gebaute unveränderte Abhängigkeiten und inkrementelle Kompilierung liefern typischerweise mehr als die Parallelisierung dessen, was bereits da ist.
- **Behandeln Sie die Pipeline-Dauer als Defekt** mit einem festgelegten Budget. Ohne festgelegte Grenze wächst die Build-Zeit monoton, weil jede einzelne Ergänzung gerechtfertigt ist und keine einzelne Ergänzung abgelehnt wird.
- Geben Sie Entwicklern einen Weg, **eine Änderung schnell gegen realistisches Verhalten zu prüfen** — einen lokalen Stub des externen Dienstes, einen aufgezeichneten Antwortsatz, einen kleinen anonymisierten Datensatz. Wo der einzige Weg herauszufinden, ob etwas funktioniert, das Deployen und Warten ist, ist dieses Warten die eigentlichen Kosten.
- **Berichten Sie die Zahlen zusammen mit Liefermetriken.** Build-Zeit ist eines der wenigen Engineering-Maße, dessen Verbesserung direkt mit Durchsatz verknüpft werden kann, was sie ungewöhnlich leicht rechtfertigbar macht — aber nur, wenn jemand darüber berichtet.

## Tradeoffs ⇄

> Die Verkürzung des Loops verstärkt sich auf alles andere, aber die Arbeit ist für Stakeholder unsichtbar, und die Aufteilung der schnellen Suite tauscht etwas Sicherheit gegen Geschwindigkeit.

**Vorteile:**

- Jede iterative Praxis wird praktikabel. Testgetriebene Entwicklung, inkrementelles Refactoring und kleine Batches sind keine Disziplinen, die einem Team fehlen — es sind Disziplinen, die ein langsamer Loop unpraktikabel macht.
- Batch-Größen fallen von selbst, weil der Grund zu bündeln — die Amortisation von Kosten — nicht mehr existiert.
- Debugging verschiebt sich von Überlegung zu Experiment, was in einem System, das niemand vollständig versteht, schneller und zuverlässiger ist.
- Kontextwechsel nehmen ab, da die Wartezeit zu kurz ist, um den Beginn von etwas anderem zu rechtfertigen.
- Die gemessene Verbesserung ist direkt zurechenbar, was sie zu einer der leichter zu verteidigenden Engineering-Investitionen macht.

**Kosten und Risiken:**

- Die Arbeit liefert kein Feature, und in einem lieferdruckgeprägten Umfeld braucht sie bewussten Schutz, um überhaupt stattzufinden.
- Die Aufteilung der Suite bedeutet, dass der schnelle Loop nicht mehr alles abdeckt, und eine Defektklasse, die in die langsame Suite wandert, wird später gefangen.
- Optimierung kann unbegrenzten Aufwand verbrauchen. Ab einem gewissen Punkt ist die verbleibende Zeit strukturell, und weitere Investition bringt wenig ein.
- Testparallelisierung und geteilte Fixtures führen bei unvollkommener Isolation zu eigener Flakiness, was den Loop schneller und weniger vertrauenswürdig machen kann.
- Schnelle lokale Umgebungen, die auf Stubs basieren, können vom Produktionsverhalten abweichen und Defekte von der Entwicklung in die Integration verschieben.

## How It Could Be

Ein achtköpfiges Team, das eine Logistikplattform pflegte, hatte eine 34-minütige Pipeline und eine lokale Testsuite, die 11 Minuten zum Start brauchte, weil sie bei jedem Lauf ein vollständiges Datenbankschema bereitstellte. Sie maßen die Loops und fanden, dass der innerste — eine Zeile ändern, den relevanten Test ausführen — durchschnittlich 13 Minuten dauerte. Niemand führte deshalb Tests lokal aus. Über ein Quartal gingen sie drei Dinge an: eine wiederverwendbare containerisierte Datenbank-Fixture, die Aufteilung von 2.400 Tests in eine 90-Sekunden-Schnellsuite und eine langsamere Integrationssuite sowie das Caching unveränderter Abhängigkeits-Builds. Der innere Loop fiel auf unter 40 Sekunden. Zur Testpraxis wurde nichts vorgeschrieben, aber die Zahl der pro Monat geschriebenen Tests verdreifachte sich in den folgenden beiden Quartalen etwa, und der Anteil der Änderungen, die die Pipeline brachen, fiel um mehr als die Hälfte, weil Entwickler diese Brüche nun vor dem Push fanden.

Die Flakiness-Arbeit brachte einen zusätzlichen Effekt, den das Team nicht erwartet hatte. Ihre Suite hatte 19 Tests, die zeitweise fehlschlugen, und die Gewohnheit des Teams war, die Pipeline erneut laufen zu lassen, bis sie bestand — eine Praxis, so normalisiert, dass niemand sie als Problem bezeichnete. Sie quarantänierten alle 19 und setzten eine Zwei-Wochen-Frist, sie zu beheben oder zu löschen. Sieben waren echt kaputte Tests und wurden gelöscht. Neun hatten echte Isolationsprobleme, die behoben wurden. Drei entpuppten sich als Offenlegung einer echten Race Condition in der Cache-Invalidierung der Anwendung, die seit mindestens einem Jahr seltene, unerklärte Produktionsinkonsistenzen verursacht hatte. Das Team hatte ein echtes Defektsignal durch das erneute Ausführen des Builds unterdrückt.
