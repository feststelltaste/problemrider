---
title: Auslieferungs-Performance-Metriken
description: Nachverfolgung von Lead Time, Deployment-Häufigkeit, Change-Failure-Rate
  und Wiederherstellungszeit als Set, sodass sowohl Verbesserungen als auch Regressionen
  sichtbar werden.
category:
- Process
- Management
- Operations
problems:
- slow-development-velocity
- slow-feature-development
- long-release-cycles
- extended-cycle-times
- reduced-predictability
- planning-credibility-issues
- immature-delivery-strategy
- quality-degradation
- inefficient-processes
- high-defect-rate-in-production
- increased-time-to-market
- delayed-value-delivery
- approval-dependencies
- blame-culture
- bottleneck-formation
- extended-review-cycles
- feature-factory
- history-of-failed-changes
- micromanagement-culture
- modernization-roi-justification-failure
- negative-brand-perception
- poor-project-control
- process-design-flaws
- product-direction-chaos
- release-anxiety
- review-bottlenecks
- rushed-approvals
- short-term-focus
- uneven-work-flow
- user-trust-erosion
- difficulty-quantifying-benefits
layout: solution
lang: de
en_slug: delivery-performance-metrics
related_solutions:
- slug: baseline-measurement
  similarity: 0.75
- slug: ci-cd-pipeline
  similarity: 0.75
- slug: value-stream-mapping
  similarity: 0.7
- slug: development-environment-optimization
  similarity: 0.7
- slug: quality-ratchet
  similarity: 0.7
- slug: code-metrics
  similarity: 0.7
---

## Description

Auslieferungs-Performance-Metriken sind ein kleines Set von Kennzahlen, das gemeinsam verfolgt wird: wie lange eine Änderung vom Commit bis zur Produktion braucht, wie oft Änderungen deployt werden, welcher Anteil von ihnen ein Problem verursacht und wie lange die Wiederherstellung dauert, wenn eines auftritt. Ihr Wert entsteht daraus, ein Set zu sein statt einzelne Zahlen, weil jede allein trivial ausgetrickst werden kann und die vier sich gegenseitig einschränken. Häufiger zu deployen sieht gut aus, bis die Fehlerrate damit ansteigt; schneller auszuliefern sieht gut aus, bis die Wiederherstellungszeit explodiert. Zusammen beschreiben sie Durchsatz und Stabilität gleichzeitig, was der Tradeoff ist, den jede Auslieferungsentscheidung tatsächlich trifft. Für Legacy-Teams liegt der praktische Nutzen in zweierlei: Sie geben eine ehrliche Baseline dafür, wie die aktuelle Situation dem entspricht, was behauptet wird, und sie machen die Wirkung von Verbesserungsarbeit in Begriffen sichtbar, die das Management bereits versteht.

## How to Apply ◆

> Legacy-Teams wird routinemäßig gesagt, schneller zu liefern, ohne dass jemand feststellt, wie schnell sie derzeit liefern oder was die Einschränkung ist.

- **Etablieren Sie die Baseline, bevor Sie irgendetwas verbessern**, aus Daten, die Sie bereits haben: Versionskontroll-Zeitstempel, Deployment-Aufzeichnungen, Vorfall-Tickets. Drei bis sechs Monate Historie zu rekonstruieren ist meist in wenigen Tagen möglich und weit mehr wert, als heute mit dem Messen zu beginnen.
- **Messen Sie die Lead Time vom Commit bis zur Produktion**, nicht von der Ticket-Erstellung. Das Commit-zu-Produktion-Intervall ist das, was der Auslieferungsprozess kontrolliert; die Zeit einzubeziehen, die eine Idee in einem Backlog lag, misst Priorisierung, was ein anderes Problem mit einer anderen Korrektur ist.
- **Verfolgen Sie alle vier zusammen und berichten Sie sie zusammen.** Ein Dashboard, das nur die Deployment-Häufigkeit zeigt, lädt dazu ein, sie isoliert zu optimieren, was dazu führt, dass ein Team öfter ausliefert und öfter bricht und das Verbesserung nennt.
- **Nutzen Sie Verteilungen, nicht Durchschnitte.** Eine mediane Lead Time von zwei Tagen mit einem 95. Perzentil von sechs Wochen beschreibt einen Prozess mit einem ernsten Problem, das der Durchschnitt verschleiert. Der Schwanz beherbergt meist die interessante Ursache.
- **Definieren Sie die Change-Failure-Rate konkret** — ein Deployment, das einen Hotfix, Rollback erfordert oder einen Vorfall verursacht — und wenden Sie die Definition konsistent an. Präzision zählt weniger als Stabilität, weil der Trend das Informative ist.
- **Berichten Sie auf Teamebene und nie auf individueller Ebene.** Diese Kennzahlen beschreiben ein Auslieferungssystem, und auf Individuen angewandt messen sie Commit-Granularität und erzeugen sofortige Manipulation.
- **Paaren Sie die Kennzahlen mit der Wertstromkarte.** Die Kennzahlen sagen Ihnen, dass der Prozess langsam ist; die Karte sagt Ihnen, wo. Kennzahlen ohne die Karte erzeugen Ermahnung, und Karten ohne Kennzahlen erzeugen Verbesserungen, die niemand bestätigen kann.
- **Ordnen Sie Verbesserungsarbeit ihnen zu.** Eine Build-Zeit-Reduktion, eine Umgebungsautomatisierung oder eine Review-Prozess-Änderung sollte sich in Lead Time oder Deployment-Häufigkeit zeigen. Verbesserungsarbeit, die keine der vier bewegt, verdient eine Erklärung.
- **Jagen Sie keinen Branchenbenchmarks nach.** Ein Team, das ein Mainframe-Batch-System pflegt, wird nicht die Deployment-Häufigkeit eines Web-Service erreichen, und den Benchmark statt den Trend zu verfolgen erzeugt Demoralisierung und Verzerrung.
- **Fügen Sie Stabilitätsmaße für Legacy-Kontexte hinzu**: den Anteil der Kapazität, der in ungeplante Arbeit fließt, und Vorfallstunden. In wartungsdominierten Umgebungen beschreiben diese die Einschränkung oft besser als Durchsatz.

## Tradeoffs ⇄

> Die vier Kennzahlen geben ein ehrliches, vergleichbares Bild der Auslieferung und machen Verbesserung nachweisbar, aber jede Kennzahl, die bestimmt, wie ein Team beurteilt wird, wird irgendwann direkt optimiert.

**Vorteile:**

- Verbesserungsarbeit wird in Begriffen nachweisbar, die das Management bereits akzeptiert, was meist die fehlende Zutat für ihre Finanzierung ist.
- Regressionen kommen früh ans Licht. Eine langsam steigende Lead Time oder Fehlerrate ist im Trend sichtbar, lange bevor sie zu einer offensichtlichen Krise wird.
- Der Durchsatz-Stabilität-Tradeoff wird explizit gemacht, was verhindert, dass eines auf Kosten des anderen verfolgt wird.
- Behauptungen über Performance — vom Team, vom Management, von Anbietern — werden überprüfbar statt rhetorisch.
- Perzentil-Berichterstattung offenbart den langen Schwanz, wo sich die strukturellen Blockierer in einem Legacy-Auslieferungsprozess meist verstecken.

**Kosten und Risiken:**

- Kennzahlen werden zu Zielen und dann manipuliert, typischerweise durch feineres Aufteilen von Änderungen oder Umklassifizieren von Fehlern. Dies ist nicht verhinderbar, nur erkennbar.
- Instrumentierung braucht echten Aufwand, wo Deployments manuell sind und Vorfälle informell verfolgt werden, was in Legacy-Umgebungen üblich ist.
- Auf Individuen angewandt oder vergleichend zwischen Teams mit sehr unterschiedlichen Systemen genutzt, richten die Kennzahlen aktiven Schaden an.
- Benchmarks laden zu unpassendem Vergleich ein, und ein Team, das an Zahlen gemessen wird, die nur mit einer anderen Architektur erreichbar sind, wird die Zahl statt das System optimieren.
- Die vier Kennzahlen sagen nichts darüber aus, ob das Gelieferte es wert ist, geliefert zu werden, und ein Team kann alle vier verbessern, während es Features ausliefert, die niemand nutzt.

## How It Could Be

Ein Team, das eine Versicherungszeichnungsplattform pflegte, stand unter anhaltendem Druck, schneller zu liefern, ohne Einigung darüber, wie schnell sie derzeit waren. Sie rekonstruierten sechs Monate Historie aus Versionskontrolle und Deployment-Logs: mediane Commit-zu-Produktion-Lead-Time von 19 Tagen, 95. Perzentil von 71 Tagen, Deployments zweimal im Monat, Change-Failure-Rate um 22 Prozent und mediane Wiederherstellungszeit von knapp über fünf Stunden. Das 95. Perzentil war der Befund. Untersuchung zeigte, dass der Schwanz fast ausschließlich aus Änderungen bestand, die ein Subsystem betrafen, dessen Deployment Koordination mit dem Release-Zeitplan eines Partners erforderte. Das war eine spezifische, adressierbare Einschränkung, die keine Menge an allgemeinem Druck, schneller zu arbeiten, je gefunden hätte. Einen unabhängigen Deployment-Pfad für dieses Subsystem auszuhandeln dauerte ein Quartal und senkte das 95. Perzentil auf 12 Tage.

Die Vier-Kennzahlen-Disziplin verhinderte im folgenden Jahr einen Fehler. Ein Vorstoß hin zu häufigerem Deployment bewegte das Team über zwei Quartale von zweimal monatlich auf zweimal wöchentlich, was wie klarer Erfolg aussah. Die parallel verfolgte Change-Failure-Rate stieg im selben Zeitraum von 22 auf 34 Prozent, und die mediane Wiederherstellungszeit verlängerte sich. Das Set zusammen zu lesen zeigte, dass das Team die Deployment-Häufigkeit erhöht hatte, indem es dieselben großen Änderungen öfter deployte, statt Änderungen kleiner zu machen, sodass jedes Deployment vergleichbares Risiko trug und es jetzt mehr davon gab. Sie pausierten den Häufigkeitsvorstoß, verbrachten ein Quartal mit Batch-Größe und der Testsuite und nahmen dann wieder auf. Am Ende des folgenden Quartals deployten sie dreimal wöchentlich mit einer Fehlerrate von 11 Prozent — ein Ergebnis, das reine Deployment-Häufigkeitsberichterstattung ein Jahr früher fälschlich als erreicht erklärt hätte.
